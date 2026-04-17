# Chizen

A path tracer written in Rust, implementing the algorithm explained at [PBR without the (brdf/pdf)](http://nihalkenkre.github.io/pbr_without_brdf_pdf). 

It is the prelude to [Satori](http://www.github.com/nihalkenkre/satori), and a successor to [Chizen](http://www.github.com/nihalkenkre/chizen).

Execute `cargo r --release` on the command line in the root folder.

## Components
### GUI
Lets the user select the file to display, controls for sample count, max bounces, and output image size.

### Viewport
Displays the current scene. Currently the scene is merged into one big vertex block and rendered through the mesh shader extension. So very large scenes crash the app.

A better approach would be to batch the mesh shader calls as indirect draw calls.

### Renderer
Progressively raytraces the scenes using the raytracing API, and Slang shaders, writing the output of each sample to an image. It has a brute force mode and a Direct + Indirect mode. 

### Display
Contains a quad to which the render target of the raytracer is mapped to. It can be zoomed into and panned around using the mouse.

### Vulkan Interface
Collection of the most commonly used vulkan objects e.g. device, memory allocators, transfer batches.

### Vulkan Objects
Interfaces around the raw vulkan handles, and a few high level operation oriented objects.

`FrameObjects` deal with handling semaphores, frame in flight, for a typical vulkan drawing session.

`DataHelper` helps to batch transfer operations and submit them together. It optionally waits on other operations through incoming semaphores.

`DataConverter` is used to convert the incoming image data into RGBA format on the GPU, and optionally converts from sRGB to linear.

## Brute Force vs Blender
| | Brute Force | Blender |
|-|-------------|---------|
| Breakfast Room 0 | [link](screenshots/0_0.jpg) | [link](screenshots/0_0_blender.jpg) |
| Breakfast Room 1 | [link](screenshots/0_1.jpg) | [link](screenshots/0_1_blender.jpg) |
| Breakfast Room 2 | [link](screenshots/0_2.jpg) | [link](screenshots/0_2_blender.jpg) |
| Cornell Box | [link](screenshots/1.jpg) | [link](screenshots/1_blender.jpg) |
| Dragon Metal and Glass | [link](screenshots/2.jpg) | [link](screenshots/2_blender.jpg) |
| Emissive Test | [link](screenshots/3.jpg) | [link](screenshots/3_blender.jpg) |
| Glass and Candle | [link](screenshots/4.jpg) | [link](screenshots/4_blender.jpg) |
| Living Room | [link](screenshots/5.jpg) | [link](screenshots/5_blender.jpg) |
| Mosquito in Amber | [link](screenshots/6.jpg) | [link](screenshots/6_blender.jpg) |
| Transmission Test | [link](screenshots/7.jpg) | [link](screenshots/7_blender.jpg) |
| Transmission Thin Wall | [link](screenshots/8.jpg) | [link](screenshots/8_blender.jpg) |

## References
[Raytracing in One Weekend](http://raytracing.github.io)  
[Physically Based Rendering](http://www.pbr-book.org/4ed/contents)  
[This answer on Stack Exchange](https://computergraphics.stackexchange.com/questions/5152/progressive-path-tracing-with-explicit-light-sampling/5153#5153)  
[Sampling Microfacet BRDF](https://agraphicsguynotes.com/posts/sample_microfacet_brdf/)  
[Shaders Monthly Series](https://www.youtube.com/playlist?list=PL8vNj3osX2PzZ-cNSqhA8G6C1-Li5-Ck8)  
[Sascha Willems Examples](https://github.com/SaschaWillems/Vulkan)  
[Vulkan + Mesh Shaders + Slang](https://medium.com/@williscool/task-and-mesh-shaders-a-practical-guide-vulkan-and-slang-25baebe6388e)