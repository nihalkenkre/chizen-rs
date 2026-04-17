#define MAX_VERTICES_PER_PRIMITIVE 64
#define MAX_TRIANGLES_PER_PRIMITIVE 124

#include "common.slang"

struct Meshlet
{
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
}

struct PushConstants
{
    CameraInstance *camera_instances;
    float4x4 *model_matrices;
    PunctualLight *punctual_lights;
    MeshEmitter *mesh_emitters;
    Material *materials;

    float3 *positions;
    Meshlet *meshlets;
    uint *meshlets_triangles;
    uint *meshlets_vertices;
    VertexData *vertices_data;

    uint camera_index;
    uint punctual_lights_count;
    uint mesh_emitters_count;
    uint meshlets_count;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct MeshOutput
{
    float4 position : SV_POSITION;
    float3 color;
    float3 t;
    float3 b;
    float3 n;
    float2 uv;
    uint material_index;
};

struct TaskShaderPayload
{
    uint meshlet_ids[32];
}

groupshared TaskShaderPayload payload;

[shader("task")]
[numthreads(32, 1, 1)]
void task_main(uint gid: SV_GroupID, uint gtid: SV_GroupThreadID, uint dtid: SV_DispatchThreadID)
{
    payload.meshlet_ids[gtid] = dtid;
    DispatchMesh(32, 1, 1, payload);
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void mesh_main(
    uint gid: SV_GroupID,
    uint gtid: SV_GroupThreadID,
    out indices uint3 triangles[MAX_TRIANGLES_PER_PRIMITIVE],
    out vertices MeshOutput vertices[MAX_VERTICES_PER_PRIMITIVE])
{
    if (payload.meshlet_ids[gid] >= pc.meshlets_count)
    {
        return;
    }

    Meshlet m = pc.meshlets[payload.meshlet_ids[gid]];

    SetMeshOutputCounts(m.vertex_count, m.triangle_count);

    if (gtid < m.triangle_count)
    {
        uint tmp_index = pc.meshlets_triangles[m.triangle_offset + gtid];

        uint index_0 = (tmp_index & 0xFF);
        uint index_1 = (tmp_index & 0xFF00) >> 8;
        uint index_2 = (tmp_index & 0xFF0000) >> 16;

        triangles[gtid] = uint3(index_0, index_1, index_2);
    }

    // since MAX_VERTICES is less than MAX_TRIANGLE_COUNT, gtid will always cover all the vertices, and more
    if (gtid < m.vertex_count)
    {
        uint vertex_index = m.vertex_offset + gtid;
        VertexData vertex_data = pc.vertices_data[vertex_index];

        float4x4 model_matrix = pc.model_matrices[vertex_data.model_matrix_material_index[0]];
        vertices[gtid].position = mul(
            pc.camera_instances[pc.camera_index].view_projection_matrix,
            mul(model_matrix, float4(pc.positions[vertex_index], 1)));

        float3 t = (vertex_data.tangent * vertex_data.tangent.w).xyz;
        float3 n = vertex_data.normal.xyz;
        t = normalize(t - dot(t, n) * n);
        float3 b = cross(n, t);

        t = normalize(mul(model_matrix, float4(t, 0)).xyz);
        b = normalize(mul(model_matrix, float4(b, 0)).xyz);
        n = normalize(mul(model_matrix, float4(n, 0)).xyz);

        vertices[gtid].t = t;
        vertices[gtid].b = b;
        vertices[gtid].n = n;
        vertices[gtid].uv = pc.vertices_data[vertex_index].uv_0.xy;
        vertices[gtid].material_index = vertex_data.model_matrix_material_index[1];
        vertices[gtid].color = float3(float(gid & 1), float(gid & 3) / 4,
                                      float(gid & 7) / 8);
    }
}

[[vk::binding(0, 0)]]
Sampler2D textures[];

[shader("fragment")]
float4 fragment_main(MeshOutput in)
{
    Material curr_mat = pc.materials[in.material_index];

    uint basecolor_index = curr_mat.basecolor_normal_metalrough_emissive_index[0];
    uint normal_index = curr_mat.basecolor_normal_metalrough_emissive_index[1];
    uint metalrough_index = curr_mat.basecolor_normal_metalrough_emissive_index[2];
    uint emissive_index = curr_mat.basecolor_normal_metalrough_emissive_index[3];

    float4 base_color = curr_mat.basecolor_factor * textures[NonUniformResourceIndex(basecolor_index)].Sample(in.uv);
    float3 normal = in.n;

    if (curr_mat.basecolor_normal_metalrough_emissive_index[1] > 0)
    {
        normal = mul(transpose(float3x3(in.t, in.b, in.n)), textures[NonUniformResourceIndex(normal_index)].Sample(in.uv).xyz * 2 - 1);
    }

    float2 metalrough = float2(curr_mat.metal_rough_transmission_ior_factor[0] * curr_mat.metal_rough_transmission_ior_factor[1]) * textures[NonUniformResourceIndex(metalrough_index)].Sample(in.uv).zy;
    float3 emission = curr_mat.emissive_factor.xyz * textures[NonUniformResourceIndex(emissive_index)].Sample(in.uv).xyz;

    float4 final_color = float4(base_color.xyz + emission, base_color.w);

    return final_color;
}
