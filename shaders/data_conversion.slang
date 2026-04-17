#include "common.slang"

struct PushConsants
{
    uint8_t *input_r8;
    uint8_t2 *input_rg8;
    uint8_t3 *input_rgb8;
    uint8_t4 *input_rgba8;

    uint image_width;
    uint image_height;
    uint image_channels;
    uint output_image_index;

    bool srgb_to_linear;
}

uint pixel_index(uint3 thread_id, uint image_width)
{
    return thread_id.y * image_width + thread_id.x;
}

[[vk::push_constant]]
ConstantBuffer<PushConsants> pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> output_rgba8[];

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 thread_id: SV_DispatchThreadID)
{
    if (thread_id.x >= pc.image_width || thread_id.y >= pc.image_height)
    {
        return;
    }

    uint pixel_index = pixel_index(thread_id, pc.image_width);

    float4 input_color = float4(0.f);

    if (pc.image_channels == 1)
    {
        input_color = float4(pc.input_r8[pixel_index], 0, 0, 255) / 255;
    }
    else if (pc.image_channels == 2)
    {
        input_color = float4(pc.input_rg8[pixel_index], 0, 255) / 255;
    }
    else if (pc.image_channels == 3)
    {
        input_color = float4(pc.input_rgb8[pixel_index], 255) / 255;
    }
    else if (pc.image_channels == 4)
    {
        input_color = float4(pc.input_rgba8[pixel_index]) / 255;
    }

    if (pc.srgb_to_linear)
    {
        output_rgba8[pc.output_image_index][thread_id.xy] = srgb_to_linear(input_color);
    }
    else
    {
        output_rgba8[pc.output_image_index][thread_id.xy] = input_color;
    }
}
