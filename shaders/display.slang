struct VSInput
{
    float2 pos;
    float2 uv;
}

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv;
}

struct PushConstants
{
    float2 pos_offset;
    float zoom_level;
}

[[vk::push_constant]]
PushConstants pc;

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out = {};

    out.pos = float4((in.pos * pc.zoom_level) - pc.pos_offset, 0, 1);
    out.uv = in.uv;

    return out;
}

[[vk::binding(0, 0)]]
Sampler2D<float4> final_render;

[shader("fragment")]
float4 fragment_main(VSOutput in)
{
    return final_render.Sample(in.uv);
}
