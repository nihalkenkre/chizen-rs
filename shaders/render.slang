#include "common.slang"

#define MESH_MISS 0
#define MESH_HIT 0
#define LIGHT_MISS 1
#define LIGHT_HIT 1
#define RAY_TYPES 2

struct PushConstants
{
    CameraInstance *camera_instances;
    PunctualLight *punctual_lights;
    MeshEmitter *mesh_emitters;
    Material *materials;

    uint4 *random_states;
    uint curr_sample;

    // 0 - Direct + Indirect
    // 1 - Brute Force (No directional light)
    uint render_mode;
    uint camera_index;
    uint punctual_light_count;
    uint mesh_emitter_count;
    uint bounce_count;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> accum_target;

[[vk::binding(1, 0)]]
RWTexture2D<float4> final_target;

[[vk::binding(2, 0)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(3, 0)]]
Sampler2D<float4> textures[];

struct Payload
{
    float3x3 tbn;
    float4 base_color;
    float3 position;
    float3 prev_position;
    float3 normal;
    float3 emission;
    float2 uv;
    float metal;
    float rough;
    float transmission;
    float ior;
    float ray_t;
    float triangle_area;
    uint material_index;
    int mesh_emitter_index;
    bool is_hit;
    bool prev_hit_specular;
}

struct LightPayload
{
    float3 hit_pos;
    float ray_t;
    uint light_index;
    uint bounce;
    bool is_hit;
}

struct HGSBTRecord
{
    VertexData *vertices_data;
    uint3 *indices;
    uint material_index;
    uint light_index;
    int mesh_emitter_index;
}

struct Attributes
{
    float2 bary;
}

[[vk::shader_record]]
ConstantBuffer<HGSBTRecord> hg_sbt;

void process_mesh_hit(Payload p, inout RayDesc ray_desc, inout float4 throughput, inout uint rand_index)
{
    uint index_0 = HybridTausUINT(rand_index, pc.random_states);
    float is_metal = Halton(5, index_0);

    uint index_1 = HybridTausUINT(rand_index, pc.random_states);
    float is_rough = Halton(7, index_1);

    uint index_2 = HybridTausUINT(rand_index, pc.random_states);
    float is_fresnel = Halton(3, index_2);

    uint index_3 = HybridTausUINT(rand_index, pc.random_states);
    float is_transmitting = Halton(2, index_3);

    float3 v = -ray_desc.Direction;
    Sample sample = TRGGXSampleHemisphere(p.tbn, v, pc.random_states, rand_index, p.rough);
    float3 l = sample.wi;

    if (p.metal >= is_metal)
    {
        float fresnel = FresnelMetallic(l, v);
        if (fresnel < is_fresnel)
        {
            throughput.xyz *= p.base_color.xyz;
        }

        ray_desc.Direction = l;
    }
    else
    {
        float fresnel = FresnelDielectric(p.normal, v, p.ior);
        if (fresnel <= is_fresnel)
        {
            if (p.transmission >= is_transmitting)
            {
                Sample sample = TRGGXSampleTransmission(p.normal, v, p.ior, pc.random_states, rand_index, p.rough);

                throughput.xyz *= p.base_color.xyz;
                ray_desc.Direction = sample.wi;
            }
            else
            {
                Sample sample = CosineSampleHemisphere(p.tbn, pc.random_states, rand_index);
                throughput.xyz *= p.base_color.xyz;

                ray_desc.Direction = sample.wi;
            }
        }
        else
        {
            ray_desc.Direction = l;
        }
    }

    ray_desc.Origin = p.position;
    ray_desc.TMin = 0.001;
    ray_desc.TMax = 10000;
}

void process_light_hit(LightPayload lp, float3 hit_point, float3 hit_normal, inout RayDesc ray_desc, inout float4 throughput, inout float4 final_color, uint4 *random_states, inout uint rand_index)
{
    PunctualLight light = pc.punctual_lights[lp.light_index];

    if (light.type == 1) // point
    {
        final_color += throughput * light.power;
    }
    else if (light.type == 2) // spot
    {
        float3 point_in_light = PointInSphere(light.position.xyz, light.icacos_ocacos_radius[2], random_states, rand_index);
        float cos_x = dot(normalize(hit_point - point_in_light), light.direction.xyz);

        float cos_inner_cone = light.icacos_ocacos_radius[0];
        float cos_outer_cone = light.icacos_ocacos_radius[1];

        float ratio = saturate((cos_x - cos_outer_cone) / (cos_inner_cone - cos_outer_cone));

        final_color += throughput * light.power * ratio;
    }
}

void process_directional_light(float3 hit_point, float3 hit_normal, inout float4 throughput, inout float4 final_color)
{
    for (uint l = 0; l < pc.punctual_light_count; ++l)
    {
        PunctualLight light = pc.punctual_lights[l];

        if (light.type == 0)
        {
            RayDesc ray_desc = {
                hit_point,
                0.001,
                -light.direction.xyz,
                10000
            };

            Payload p = {};

            TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, ray_desc, p);

            if (!p.is_hit)
            {
                final_color += throughput * light.power * max(dot(hit_normal, ray_desc.Direction), 0);
            }
        }
    }
}

[shader("raygeneration")]
void raygen()
{
    uint3 launch_id = DispatchRaysIndex();
    uint3 launch_size = DispatchRaysDimensions();
    uint rand_index = launch_id.y * launch_size.x + launch_id.x;

    const float2 pixel_center = float2(launch_id.xy) + float2(0.5, 0.5);
    const float2 in_uv = pixel_center / float2(launch_size.xy);

    CameraInstance cam_info = pc.camera_instances[pc.camera_index];

    uint index_0 = HybridTausUINT(rand_index, pc.random_states);
    uint index_1 = HybridTausUINT(rand_index, pc.random_states);

    float2 d_offset = (float2(Halton(2, index_0), Halton(3, index_1)) * 2 - 1) / (launch_size.xy);
    float2 d = in_uv * 2.f - 1.f;// + d_offset;
    float4 target = mul(cam_info.projection_inverse_matrix, float4(d.x, d.y, 1, 1));

    RayDesc ray_desc;
    ray_desc.Origin = mul(cam_info.view_inverse_matrix, float4(0, 0, 0, 1)).xyz;
    ray_desc.Direction = mul(cam_info.view_inverse_matrix, float4(normalize(target.xyz), 0)).xyz;
    ray_desc.TMin = cam_info.z_near_far[0];
    ray_desc.TMax = cam_info.z_near_far[1];

    Payload p = {};
    float4 throughput = float4(1, 1, 1, 1);
    float4 final_color = float4(0, 0, 0, 1);

    LightPayload lp = {};

    bool mesh_miss = false;
    bool light_miss = false;

    if (pc.render_mode == 0) // NEE + Indirect
    {
        RayDesc indirect_ray = ray_desc;
        for (uint b = 0; b <= pc.bounce_count; ++b)
        {
            TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, indirect_ray, p);

            // if (p.is_hit)
            // {
            //     uint32_t index_0 = HybridTausUINT(rand_index, pc.random_states);
            //     float is_transmitting = Halton(2, index_0);

            //     if (p.transmission > is_transmitting)
            //     {
            //         float3 wi = TRGGXSampleTransmission(p.normal, -indirect_ray.Direction, p.ior, pc.random_states, rand_index, p.rough).wi;
            //         final_color = srgb_to_linear(float4(abs(wi), 1));
            //         indirect_ray.Direction = wi;
            //     }
            //     indirect_ray.Origin = p.position;

            // }

            Sample cos_sample = CosineSampleHemisphere(p.tbn, pc.random_states, rand_index);

            if (p.is_hit)
            {
                if (b == 0 || p.prev_hit_specular)
                {
                    final_color.xyz += p.emission * throughput.xyz;
                }

                uint index_0 = HybridTausUINT(rand_index, pc.random_states);
                float is_metal = Halton(5, index_0);

                uint index_2 = HybridTausUINT(rand_index, pc.random_states);
                float is_fresnel = Halton(3, index_2);

                float3 v = -indirect_ray.Direction;
                Sample sample = TRGGXSampleHemisphere(p.tbn, v, pc.random_states, rand_index, p.rough);
                float3 l = sample.wi;

                if (p.metal >= is_metal)
                {
                    float fresnel = FresnelMetallic(l, v);

                    if (fresnel < is_fresnel)
                    {
                        throughput.xyz *= p.base_color.xyz;
                    }

                    p.prev_hit_specular = true;
                    indirect_ray.Direction = l;
                }
                else
                {
                    float fresnel = FresnelDielectric(p.normal, v, p.ior);

                    if (fresnel < is_fresnel)
                    {
                        uint index_3 = HybridTausUINT(rand_index, pc.random_states);
                        float is_transmitting = Halton(2, index_3);

                        if (p.transmission >= is_transmitting)
                        {
                            Sample sample = TRGGXSampleTransmission(p.normal, v, p.ior, pc.random_states, rand_index, p.rough);

                            p.prev_hit_specular = true;
                            throughput.xyz *= p.base_color.xyz;
                            indirect_ray.Direction = sample.wi;
                        }
                        else
                        {
                            Sample cos_sample = CosineSampleHemisphere(p.tbn, pc.random_states, rand_index);
                            indirect_ray.Direction = cos_sample.wi;

                            float3 light_color = {};

                            for (uint l = 0; l < pc.punctual_light_count; ++l)
                            {
                                PunctualLight light = pc.punctual_lights[l];

                                if (light.type == 0)
                                {
                                    float3 point_in_light = PointInSphere(p.position - light.direction.xyz * 10000, light.icacos_ocacos_radius[2], pc.random_states, rand_index);
                                    RayDesc light_ray = {
                                        p.position,
                                        0.001,
                                        normalize(point_in_light - p.position),
                                        10000
                                    };

                                    Payload lp = {};
                                    TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, light_ray, lp);

                                    if (!lp.is_hit)
                                    {
                                        float NdotI = saturate(dot(p.normal, -light.direction.xyz));
                                        light_color += light.power.xyz * p.base_color.xyz * NdotI;
                                    }
                                }
                                else if (light.type == 1)
                                {
                                    float3 point_in_light = PointInSphere(light.position.xyz, light.icacos_ocacos_radius[2], pc.random_states, rand_index);
                                    float distance = length(point_in_light - p.position);
                                    float NdotI = saturate(dot(normalize(point_in_light - p.position), p.normal));

                                    RayDesc light_ray = {
                                        p.position,
                                        0.001,
                                        normalize(point_in_light - p.position),
                                        distance
                                    };

                                    Payload lp = {};
                                    TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, light_ray, lp);

                                    if (!lp.is_hit)
                                    {
                                        light_color += (light.power.xyz * NdotI * p.base_color.xyz) / (distance * distance);
                                    }
                                }
                                else if (light.type == 2)
                                {
                                    float3 point_in_light = PointInSphere(light.position.xyz, light.icacos_ocacos_radius[2], pc.random_states, rand_index);
                                    float distance = length(point_in_light - p.position);

                                    RayDesc light_ray = {
                                        p.position + 0.001 * -indirect_ray.Direction,
                                        0.001,
                                        normalize(point_in_light - p.position),
                                        distance
                                    };

                                    Payload lp = {};
                                    TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, light_ray, lp);

                                    if (!lp.is_hit)
                                    {
                                        float cos_x = dot(normalize(p.position - point_in_light), light.direction.xyz);

                                        float cos_inner_cone = light.icacos_ocacos_radius[0];
                                        float cos_outer_cone = light.icacos_ocacos_radius[1];

                                        float ratio = saturate((cos_x - cos_outer_cone) / (cos_inner_cone - cos_outer_cone));

                                        if (ratio > 0.001)
                                        {
                                            float NdotI = saturate(dot(normalize(point_in_light - p.position), p.normal));
                                            light_color += (light.power.xyz * ratio * NdotI * p.base_color.xyz) / (distance * distance);
                                        }
                                    }
                                }
                            }

                            for (uint m = 0; m < pc.mesh_emitter_count; ++m)
                            {
                                if (p.mesh_emitter_index == m)
                                {
                                    continue;
                                }

                                MeshEmitter mesh = pc.mesh_emitters[m];
                                float3 point_in_mesh = PointInBoundingBox(mesh.bounding_box_min, mesh.bounding_box_max, pc.random_states, rand_index);

                                float distance = length(point_in_mesh - p.position);

                                Payload pl = {};
                                RayDesc light_ray = {
                                    p.position,
                                    0.001,
                                    normalize(point_in_mesh - p.position),
                                    distance
                                };

                                TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, light_ray, pl);

                                if (pl.is_hit)
                                {
                                    if (pl.mesh_emitter_index == m)
                                    {
                                        float NdotI = saturate(dot(p.normal, light_ray.Direction));
                                        float LdotS = saturate(dot(pl.normal, -light_ray.Direction));
                                        light_color += (pl.emission * p.base_color.xyz * NdotI * LdotS * mesh.area_avgemission[0]) / (distance * distance);
                                    }
                                }
                            }

                            final_color.xyz += throughput.xyz * light_color;
                            throughput *= p.base_color;
                        }
                    }
                    else
                    {
                        p.prev_hit_specular = true;
                        indirect_ray.Direction = l;
                    }
                }

                // final_color = srgb_to_linear(float4(p.normal, 1));
                indirect_ray.Origin = p.position;
                indirect_ray.TMin = 0.001;
                indirect_ray.TMax = 10000;
            }
            else
            {
                if (b == 0)
                {
                    final_color += throughput;
                }
                else
                {
                    final_color += throughput;// * float4(0.1, 0.1, 0.1, 1);
                }

                break; // miss
            }
        }
    }
    else if (pc.render_mode == 1) // Brute Force
    {
        for (uint b = 0; b <= pc.bounce_count; ++b)
        {
            lp.bounce = b;

            TraceRay(tlas, 0, MESH_MASK, MESH_HIT, RAY_TYPES, MESH_MISS, ray_desc, p);
            TraceRay(tlas, 0, LIGHT_MASK, LIGHT_HIT, RAY_TYPES, LIGHT_MISS, ray_desc, lp);

            if (p.is_hit && lp.is_hit)
            {
                // mesh closest hit
                if (p.ray_t < lp.ray_t)
                {
                    if (p.emission.x > 0 || p.emission.y > 0 || p.emission.z > 0)
                    {
                        final_color.xyz += throughput.xyz * p.emission;
                        break;
                    }

                    process_mesh_hit(p, ray_desc, throughput, rand_index);
                }
                // light closest hit
                else
                {
                    process_light_hit(lp, p.position, p.normal, ray_desc, throughput, final_color, pc.random_states, rand_index);

                    break;
                }
            }
            else if (p.is_hit && !lp.is_hit)
            {
                if (p.emission.x > 0 || p.emission.y > 0 || p.emission.z > 0)
                {
                    final_color.xyz += throughput.xyz * p.emission;
                    break;
                }

                process_mesh_hit(p, ray_desc, throughput, rand_index);
            }
            else if (lp.is_hit && !p.is_hit)
            {
                process_light_hit(lp, p.position, p.normal, ray_desc, throughput, final_color, pc.random_states, rand_index);

                break;
            }
            else
            {
                // if (b == 0)
                // {
                //     final_color += throughput;
                // }
                // else
                // {
                //     final_color += throughput * float4(0.1, 0.1, 0.1, 1);
                // }

                break; // missed meshes and lights
            }
        }
    }

    accum_target[launch_id.xy] += final_color;
    final_target[launch_id.xy] = accum_target[launch_id.xy] / pc.curr_sample;
}

[shader("miss")]
void mesh_miss(inout Payload p)
{
    p.is_hit = false;
}

[shader("anyhit")]
void mesh_anyhit(inout Payload p, in BuiltInTriangleIntersectionAttributes attribs)
{
    const float3 bary_coords = float3(attribs.barycentrics.x, attribs.barycentrics.y, 1.0f - attribs.barycentrics.x - attribs.barycentrics.y);

    uint3 index_triplet = *(hg_sbt.indices + PrimitiveIndex());

    float2 uv = hg_sbt.vertices_data[index_triplet.x].uv_0.xy * bary_coords.z +
                hg_sbt.vertices_data[index_triplet.y].uv_0.xy * bary_coords.x +
                hg_sbt.vertices_data[index_triplet.z].uv_0.xy * bary_coords.y;

    Material curr_mat = pc.materials[hg_sbt.material_index];
    uint basecolor_index = curr_mat.basecolor_normal_metalrough_emissive_index[0];
    float4 base_color = curr_mat.basecolor_factor * textures[NonUniformResourceIndex(basecolor_index)].SampleLevel(uv, 0);

    if (base_color.w < 0.01)
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void mesh_closesthit(inout Payload p, in BuiltInTriangleIntersectionAttributes attribs)
{
    uint3 launch_id = DispatchRaysIndex();
    uint3 launch_size = DispatchRaysDimensions();

    uint flat_index = launch_id.y * launch_size.x + launch_id.x;
    const float3 bary_coords = float3(attribs.barycentrics.x, attribs.barycentrics.y, 1.0f - attribs.barycentrics.x - attribs.barycentrics.y);

    uint3 index_triplet = *(hg_sbt.indices + PrimitiveIndex());

    float2 uv = hg_sbt.vertices_data[index_triplet.x].uv_0.xy * bary_coords.z +
                hg_sbt.vertices_data[index_triplet.y].uv_0.xy * bary_coords.x +
                hg_sbt.vertices_data[index_triplet.z].uv_0.xy * bary_coords.y;

    float4 tangent = hg_sbt.vertices_data[index_triplet.x].tangent * bary_coords.z +
                     hg_sbt.vertices_data[index_triplet.y].tangent * bary_coords.x +
                     hg_sbt.vertices_data[index_triplet.z].tangent * bary_coords.y;

    float3 n = hg_sbt.vertices_data[index_triplet.x].normal.xyz * bary_coords.z +
               hg_sbt.vertices_data[index_triplet.y].normal.xyz * bary_coords.x +
               hg_sbt.vertices_data[index_triplet.z].normal.xyz * bary_coords.y;

    float3 t = (tangent * tangent.w).xyz;
    t = normalize(t - dot(t, n) * n);

    float3 b = cross(n, t);

    t = normalize(mul(ObjectToWorld3x4(), float4(t, 0)).xyz);
    b = normalize(mul(ObjectToWorld3x4(), float4(b, 0)).xyz);
    n = normalize(mul(ObjectToWorld3x4(), float4(n, 0)).xyz);

    float3x3 tbn = transpose(float3x3(t, b, n));

    Material curr_mat = pc.materials[hg_sbt.material_index];

    uint basecolor_index = curr_mat.basecolor_normal_metalrough_emissive_index[0];
    uint normal_index = curr_mat.basecolor_normal_metalrough_emissive_index[1];
    uint metalrough_index = curr_mat.basecolor_normal_metalrough_emissive_index[2];
    uint emissive_index = curr_mat.basecolor_normal_metalrough_emissive_index[3];
    uint transmission_index = curr_mat.transmission_index[0];

    float4 base_color = curr_mat.basecolor_factor * textures[NonUniformResourceIndex(basecolor_index)].SampleLevel(uv, 0);
    float3 normal = n;
    if (normal_index != 0)
    {
        normal = mul(tbn, textures[NonUniformResourceIndex(normal_index)].SampleLevel(uv, 0).xyz * 2 - 1);
        t = normalize(t - dot(t, normal) * normal);
        b = cross(normal, t);
        tbn = transpose(float3x3(t, b, normal));
    }

    float2 metalrough = float2(curr_mat.metal_rough_transmission_ior_factor[0], curr_mat.metal_rough_transmission_ior_factor[1]) * textures[NonUniformResourceIndex(metalrough_index)].SampleLevel(uv, 0).zy;
    float3 emission = curr_mat.emissive_factor.xyz * textures[NonUniformResourceIndex(emissive_index)].SampleLevel(uv, 0).xyz;
    float transmission = curr_mat.metal_rough_transmission_ior_factor[2] * textures[NonUniformResourceIndex(transmission_index)].SampleLevel(uv, 0).x;

    float3 v0 = HitTriangleVertexPosition(0);
    float3 v1 = HitTriangleVertexPosition(1);
    float3 v2 = HitTriangleVertexPosition(2);

    float3 v1v0 = v1 - v0;
    float3 v2v0 = v2 - v0;
    p.triangle_area = length(cross(v2v0, v1v0)) / 2;

    p.tbn = tbn;
    p.base_color = base_color;
    p.prev_position = p.position;
    p.position = WorldRayOrigin() + (WorldRayDirection() * RayTCurrent());
    p.normal = normal;
    p.emission = emission;
    p.uv = uv;
    p.metal = max(metalrough.x, 0.0001);
    p.rough = max(metalrough.y, 0.0001);
    p.transmission = transmission;
    p.ior = curr_mat.metal_rough_transmission_ior_factor[3];
    p.ray_t = RayTCurrent();
    p.material_index = hg_sbt.material_index;
    p.mesh_emitter_index = hg_sbt.mesh_emitter_index;
    p.is_hit = true;
}

struct LightAttributes
{
    uint x;
}

[shader("miss")]
void light_miss(inout LightPayload lp)
{
    lp.is_hit = false;
}

[shader("intersection")]
void light_intersection()
{
    PunctualLight light = pc.punctual_lights[hg_sbt.light_index];

    if (light.type == 0)
    {
        return;
    }

    float3 oc = light.position.xyz - WorldRayOrigin();
    float a = length(WorldRayDirection()) * length(WorldRayDirection());
    float h = dot(WorldRayDirection(), oc);
    float c = length(oc) * length(oc) - light.icacos_ocacos_radius[2] * light.icacos_ocacos_radius[2];

    float dis = h * h - a * c;

    if (dis >= 0)
    {
        LightAttributes direct_attrs = {};
        ReportHit((h - sqrt(dis)) / a, 0, direct_attrs);
    }
}

[shader("closesthit")]
void light_closesthit(inout LightPayload lp, in LightAttributes direct_attrs)
{
    if (lp.bounce == 0)
    {
        return;
    }

    lp.hit_pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    lp.ray_t = RayTCurrent();
    lp.light_index = hg_sbt.light_index;
    lp.is_hit = true;
}
