#define PI 3.14159265

#define MESH_MASK 0x1 << 0
#define LIGHT_MASK 0x1 << 1

struct CameraInstance
{
    float4x4 view_projection_matrix;
    float4x4 view_inverse_matrix;
    float4x4 projection_inverse_matrix;
    float z_near_far[4];
}

struct VertexData
{
    float4 tangent;
    float4 normal;
    float2 uv_0;
    uint model_matrix_material_index[2];
}

struct Material
{
    float4 basecolor_factor;
    float4 emissive_factor;
    uint basecolor_normal_metalrough_emissive_index[4];
    float metal_rough_transmission_ior_factor[4];
    uint transmission_index[4];
}

enum PunctualLightType
{
    Direction,
    Point,
    Spot,
}

void PrintFloat2(float2 value)
{
    printf("%f %f\n", value.x, value.y);
}

void PrintFloat3(float3 value)
{
    printf("%f %f %f\n", value.x, value.y, value.z);
}

void PrintFloat4(float4 value)
{
    printf("%f %f %f %f\n", value.x, value.y, value.z, value.w);
}

float3 PointInSphere(float3 center, float radius, uint4 *random_states, inout uint rand_index)
{
    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);
    uint index_2 = HybridTausUINT(rand_index, random_states);

    float3 point = float3(
        Halton(2, index_0),
        Halton(3, index_1),
        Halton(5, index_2));

    point = point * 2 - 1;
    point *= radius;

    return point + center;
}

float3 PointInBoundingBox(float4 bbox_min, float4 bbox_max, uint4 *random_states, inout uint rand_index)
{
    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);
    uint index_2 = HybridTausUINT(rand_index, random_states);

    float3 point = float3(
        Halton(2, index_0),
        Halton(3, index_1),
        Halton(5, index_2));

    return bbox_min.xyz + point * (bbox_max.xyz - bbox_min.xyz);
}

struct PunctualLight
{
    float4 position;
    float4 direction;
    float4 scale;
    float4 power;
    float4 bbox_min;
    float4 bbox_max;
    float icacos_ocacos_radius[4];
    PunctualLightType type;
}

struct MeshEmitter
{
    float4 bounding_box_min;
    float4 bounding_box_max;
    // [0]: area, [1-2-3]: avg_emission
    float area_avgemission[4];
}

float4 srgb_to_linear(float4 srgb)
{
    return float4(
        srgb.x <= 0.04045 ? srgb.x / 12.92 : pow(((srgb.x + 0.055) / 1.055), 2.4),
        srgb.y <= 0.04045 ? srgb.y / 12.92 : pow(((srgb.y + 0.055) / 1.055), 2.4),
        srgb.z <= 0.04045 ? srgb.z / 12.92 : pow(((srgb.z + 0.055) / 1.055), 2.4),
        srgb.w);
}

float4 linear_to_srgb(float4 linear)
{
    return float4(
        linear.x <= 0.0031308 ? linear.x * 12.92 : 1.055 * pow(linear.x, 0.4166666) - 0.055,
        linear.y <= 0.0031308 ? linear.y * 12.92 : 1.055 * pow(linear.y, 0.4166666) - 0.055,
        linear.z <= 0.0031308 ? linear.z * 12.92 : 1.055 * pow(linear.z, 0.4166666) - 0.055,
        linear.w);
}

uint TausStep(inout uint z, int S1, int S2, int S3, uint M)
{
    uint b = (((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
}

uint LCGStep(inout uint z, uint A, uint C)
{
    return z = (A * z + C);
}

float HybridTaus(uint flat_index, uint4 *random_states)
{
    uint4 z = random_states[flat_index];
    float rand = 2.3283064365387e-10 * (TausStep(z.x, 13, 19, 12, 4294967294) ^
                                        TausStep(z.y, 2, 25, 4, 4294967288) ^
                                        TausStep(z.z, 3, 11, 17, 4294967280) ^
                                        LCGStep(z.w, 1664525, 1013904223));

    random_states[flat_index] = z;
    return rand;
}

uint HybridTausUINT(inout uint rand_index, uint4 *random_states)
{
    uint4 z = random_states[rand_index];
    uint rand = (TausStep(z.x, 13, 19, 12, 4294967294) ^
                 TausStep(z.y, 2, 25, 4, 4294967288) ^
                 TausStep(z.z, 3, 11, 17, 4294967280) ^
                 LCGStep(z.w, 1664525, 1013904223));

    random_states[rand_index] = z;
    return rand;
}

float Halton(uint base, inout uint index)
{
    float result = 0.0;
    float digit_weight = 1.0;

    while (index > 0)
    {
        digit_weight = digit_weight / float(base);
        uint numerator = index % base;
        result += float(numerator) * digit_weight;
        index = index / base;
    }

    return result;
}

struct Sample
{
    float3 wi;
    float theta;
}

Sample UniformSampleHemisphere(float3x3 tbn, uint4 *random_states, uint rand_index)
{
    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);

    float theta = acos(1 - Halton(2, index_0));
    float phi = 2 * PI * Halton(3, index_1);

    Sample sample = {};

    sample.wi = mul(tbn, float3(
                             sin(theta) * cos(phi),
                             sin(theta) * sin(phi),
                             cos(theta)));
    sample.theta = theta;

    return sample;
}

float UniformEval(float theta)
{
    return sin(theta);
}

float UniformPdf(float theta)
{
    return sin(theta) / (2 * PI);
}

Sample CosineSampleHemisphere(float3x3 tbn, uint4 *random_states, inout uint rand_index)
{
    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);

    float theta = asin(sqrt(Halton(2, index_0)));
    float phi = 2 * PI * Halton(3, index_1);

    Sample sample = {};

    float3 h = mul(tbn, float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));

    sample.wi = h;
    sample.theta = theta;

    return sample;
}

float CosineEval(float theta)
{
    return cos(theta);
}

float CosinePdf(float theta)
{
    return (cos(theta)) / PI;
}

Sample TRGGXSampleHemisphere(float3x3 tbn, float3 v, uint4 *random_states, uint rand_index, float roughness)
{
    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);

    float u_rand = Halton(2, index_0);
    float v_rand = Halton(3, index_1);

    float a = roughness * roughness;
    float theta = atan(a * (sqrt(u_rand / (1 - u_rand))));
    float phi = 2 * PI * v_rand;

    Sample sample = {};

    float3 h = mul(tbn, float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));

    sample.wi = reflect(-v, h);
    sample.theta = theta;

    return sample;
}

float DGGX(float3 n, float3 l, float3 v, float roughness)
{
    float3 h = normalize(l + v);

    float a2 = roughness * roughness * roughness * roughness;
    float NdotH = max(dot(n, h), 0);
    float denom = max((NdotH * NdotH) * (a2 - 1) + 1, 0.001);

    return a2 / (PI * denom * denom);
}

float GGGX(float3 n, float3 l, float3 v, float roughness)
{
    float a2 = roughness * roughness * roughness * roughness;

    float NdotV = max(dot(n, v), 0);
    float GV = (2 * NdotV) / (NdotV + sqrt(a2 + (1 - a2) * (NdotV * NdotV)));

    float NdotL = max(dot(n, l), 0);
    float GL = (2 * NdotL) / (NdotL + sqrt(a2 + (1 - a2) * (NdotL * NdotL)));

    return GV * GL;
}

float FresnelMetallic(float3 l, float3 v)
{
    float3 h = normalize(l + v);
    float v_dot_h = dot(v, h);

    if (isnan(v_dot_h))
    {
        return 0;
    }

    float avoid_pow = max(1 - dot(v, h), 0.1);

    return avoid_pow * avoid_pow * avoid_pow * avoid_pow * avoid_pow;
}

float TRGGXEvalMetallic(float3 n, float3 l, float3 v, float roughness)
{
    float D = DGGX(n, l, v, roughness);
    float G = GGGX(n, l, v, roughness);
    float F = FresnelMetallic(l, v);

    return (D * G * F) / (4 * max(dot(n, v), 0.00001) * max(dot(n, l), 0.00001));
}

float TRGGXPdf(float3 l, float3 v, float theta, float roughness)
{
    float a2 = roughness * roughness * roughness * roughness;
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);

    float nume = (2 * a2 * cos_theta * sin_theta);
    float denom = max(((a2 - 1) * (cos_theta * cos_theta) + 1), 0.00001);

    return nume / (denom * denom) / (4 * dot(l, v));
}

float TRGGXBrdfMetallic(float3 n, float3 l, float3 v, float theta, float roughness)
{
    float eval = TRGGXEvalMetallic(n, l, v, roughness);
    float pdf = TRGGXPdf(l, v, theta, roughness);

    return eval / pdf;
}

float FresnelDielectric(float3 n, float3 v, float eta)
{
    float3 i = v;

    float cos_theta_i = dot(i, n);
    if (cos_theta_i < 0)
    {
        n = -n;
        cos_theta_i = -cos_theta_i;
        eta = 1 / eta;
    }

    float sin_2_theta_i = 1 - (cos_theta_i * cos_theta_i);
    float sin_2_theta_t = sin_2_theta_i / (eta * eta);
    if (sin_2_theta_t >= 1)
    {
        return 1;
    }
    float cos_theta_t = clamp(sqrt(1 - sin_2_theta_t), -1, 1);

    float r_parl = ((eta * cos_theta_i) - cos_theta_t) / ((eta * cos_theta_i) + cos_theta_t);
    float r_perp = (cos_theta_i - (eta * cos_theta_t)) / (cos_theta_i + (eta * cos_theta_t));

    return 0.5 * ((r_parl * r_parl) + (r_perp * r_perp));
}

float TRGGXEvalDielectricSpec(float3 n, float3 l, float3 v, float roughness, float eta)
{
    float D = DGGX(n, l, v, roughness);
    float G = GGGX(n, l, v, roughness);
    float F = FresnelDielectric(v, n, eta);

    return (D * G * F) / (4 * max(dot(n, v), 0.001) * max(dot(n, l), 0.001));
}

float TRGGXBrdfDielectric(float3 n, float3 l, float3 v, float theta, float roughness, float eta)
{
    float eval = TRGGXEvalDielectricSpec(n, l, v, roughness, eta);
    float pdf = TRGGXPdf(l, v, theta, roughness);

    return eval / pdf;
}

Sample TRGGXSampleTransmission(float3 n, float3 v, float eta, uint4 *random_states, uint rand_index, float roughness)
{
    float3 i = v;

    float cos_theta_i = dot(i, n);
    if (cos_theta_i < 0)
    {
        n = -n;
        cos_theta_i = -cos_theta_i;
        eta = 1 / eta;
    }

    float sin_2_theta_i = 1 - (cos_theta_i * cos_theta_i);
    float sin_2_theta_t = sin_2_theta_i / (eta * eta);

    Sample sample = {};

    if (sin_2_theta_t >= 1)
    {
        sample.wi = reflect(-i, n);
    }
    else
    {
        float cos_theta_t = clamp(sqrt(1 - sin_2_theta_t), -1, 1);

        float3 r = -i / eta + (cos_theta_i / eta - cos_theta_t) * n;

        sample.wi = r;
    }

    float3 n = sample.wi;
    float3 t_ref = abs(n.z) > 0.9999f ? float3(0, 1, 0) : float3(0, 0, 1);

    float3 b = normalize(cross(n, t_ref));
    float3 t = cross(n, b);

    float3x3 tbn = transpose(float3x3(t, b, n));

    uint index_0 = HybridTausUINT(rand_index, random_states);
    uint index_1 = HybridTausUINT(rand_index, random_states);

    float u_rand = Halton(2, index_0);
    float v_rand = Halton(3, index_1);

    float a = roughness * roughness;
    float theta = atan(a * (sqrt(u_rand / (1 - u_rand))));
    float phi = 2 * PI * v_rand;

    float3 h = mul(tbn, float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));

    sample.wi = h;

    return sample;
}
