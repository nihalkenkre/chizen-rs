use std::{
    ffi::{CStr, c_void},
    num::NonZeroI16,
};

use ash::{
    ext,
    khr::{self},
    vk::{self},
};
use vk_mem::Alloc;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::vulkan_objects;

pub struct Instance {
    pub instance: ash::Instance,
}

impl Instance {
    pub fn new(required_extensions: &[*const i8], vulkan_api_version: u32) -> Self {
        let ash_entry = ash::Entry::linked();

        let mut req_exts = vec![];
        if cfg!(debug_assertions) {
            req_exts.push(ext::debug_utils::NAME.as_ptr());
        }

        unsafe {
            for required_extension in required_extensions.iter() {
                req_exts.push(CStr::from_ptr(*required_extension).as_ptr());
            }

            Self {
                instance: ash_entry
                    .create_instance(
                        &vk::InstanceCreateInfo::default()
                            .application_info(
                                &vk::ApplicationInfo::default().api_version(vulkan_api_version),
                            )
                            .enabled_extension_names(&req_exts),
                        None,
                    )
                    .unwrap(),
            }
        }
    }

    pub fn destroy(self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct PhysicalDeviceData<'instance_lifetime> {
    pub physical_device: vk::PhysicalDevice,

    pub graphics_queue_family_index: u32,
    pub transfer_queue_family_index: u32,
    pub compute_queue_family_index: u32,

    pub queue_family_indices: Vec<u32>,
    pub properties: vk::PhysicalDeviceProperties2<'instance_lifetime>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties2<'instance_lifetime>,
    pub raytracing_properties:
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'instance_lifetime>,
    pub acceleration_structure_properties:
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'instance_lifetime>,
}

impl<'instance_lifetime> PhysicalDeviceData<'instance_lifetime> {
    pub fn new(
        instance: &ash::Instance,
        loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        unsafe {
            let mut properties = vk::PhysicalDeviceProperties2::default();

            // find the first discrete or intergrated GPU, simple use case
            let physical_device = match instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|physical_device| {
                    instance.get_physical_device_properties2(*physical_device, &mut properties);
                    match properties.properties.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => true,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => true,
                        _ => false,
                    }
                }) {
                Some(x) => x,
                None => {
                    println!("No GPUs on the system. Exiting...");
                    vk::PhysicalDevice::null()
                }
            };

            let mut queue_family_properties =
                vec![
                    vk::QueueFamilyProperties2::default();
                    instance.get_physical_device_queue_family_properties2_len(physical_device)
                ];

            instance.get_physical_device_queue_family_properties2(
                physical_device,
                &mut queue_family_properties,
            );

            // look for queuee family with GRAPHICS_BIT set
            let graphics_queue_family_index = queue_family_properties
                .iter()
                .enumerate()
                .find(|(index, qfp)| {
                    loader
                        .get_physical_device_surface_support(
                            physical_device,
                            *index as u32,
                            *surface,
                        )
                        .unwrap()
                        && qfp
                            .queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                })
                .unwrap()
                .0 as u32;

            // look for queuee family with only TRANSFER_BIT and !GRAPHICS_BIT set
            // if no queue found then look for any queue with TRANSFER_BIT set.
            let transfer_queue_family_index =
                match queue_family_properties.iter().enumerate().find(|(_, qfp)| {
                    qfp.queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::TRANSFER)
                        && !qfp
                            .queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                }) {
                    Some(x) => x.0 as u32,
                    None => {
                        queue_family_properties
                            .iter()
                            .enumerate()
                            .find(|(_, qfp)| {
                                qfp.queue_family_properties
                                    .queue_flags
                                    .contains(vk::QueueFlags::TRANSFER)
                            })
                            .unwrap()
                            .0 as u32
                    }
                };

            // look for queuee family with only COMPUTE_BIT and !GRAPHICS_BIT set
            // if no queue found then look for any queue with COMPUTE_BIT set.
            let compute_queue_family_index =
                match queue_family_properties.iter().enumerate().find(|(_, qfp)| {
                    qfp.queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
                        && !qfp
                            .queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                }) {
                    Some(x) => x.0 as u32,
                    None => {
                        queue_family_properties
                            .iter()
                            .enumerate()
                            .find(|(_, qfp)| {
                                qfp.queue_family_properties
                                    .queue_flags
                                    .contains(vk::QueueFlags::COMPUTE)
                            })
                            .unwrap()
                            .0 as u32
                    }
                };

            let mut raytracing_properties =
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut acceleration_structure_properties =
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut memory_properties = vk::PhysicalDeviceMemoryProperties2::default();

            let mut pdpn = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut raytracing_properties)
                .push_next(&mut acceleration_structure_properties);

            instance.get_physical_device_properties2(physical_device, &mut pdpn);
            instance
                .get_physical_device_memory_properties2(physical_device, &mut memory_properties);

            Self {
                physical_device,
                graphics_queue_family_index,
                transfer_queue_family_index,
                compute_queue_family_index,
                queue_family_indices: vec![
                    graphics_queue_family_index,
                    compute_queue_family_index,
                    transfer_queue_family_index,
                ],
                properties,
                memory_properties,
                raytracing_properties,
                acceleration_structure_properties,
            }
        }
    }
}

pub struct SurfaceData<'instance_lifetime> {
    pub capabilities: vk::SurfaceCapabilities2KHR<'instance_lifetime>,
    pub surface_format: vk::SurfaceFormat2KHR<'instance_lifetime>,
    pub present_mode: vk::PresentModeKHR,
    pub surface: vk::SurfaceKHR,
    pub loader: khr::surface::Instance,
}

impl<'instance_lifetime> SurfaceData<'instance_lifetime> {
    pub fn new(
        instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        unsafe {
            let surface = ash_window::create_surface(
                &ash::Entry::linked(),
                instance,
                display_handle,
                window_handle,
                None,
            )
            .unwrap();

            Self {
                surface,
                capabilities: vk::SurfaceCapabilities2KHR::default(),
                surface_format: vk::SurfaceFormat2KHR::default(),
                present_mode: vk::PresentModeKHR::default(),
                loader: khr::surface::Instance::new(&ash::Entry::linked(), instance),
            }
        }
    }

    pub fn populate_data(&mut self, physical_device: &vk::PhysicalDevice) {
        unsafe {
            self.capabilities = vk::SurfaceCapabilities2KHR::default().surface_capabilities(
                self.loader
                    .get_physical_device_surface_capabilities(*physical_device, self.surface)
                    .unwrap(),
            );

            self.surface_format = match self
                .loader
                .get_physical_device_surface_formats(*physical_device, self.surface)
                .unwrap()
                .iter()
                .find(|&surface_format| {
                    surface_format.format == vk::Format::R8G8B8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                }) {
                Some(x) => vk::SurfaceFormat2KHR::default().surface_format(*x),
                None => vk::SurfaceFormat2KHR::default(),
            };

            self.present_mode = match self
                .loader
                .get_physical_device_surface_present_modes(*physical_device, self.surface)
                .unwrap()
                .iter()
                .find(|&present_mode| *present_mode == vk::PresentModeKHR::MAILBOX)
            {
                Some(x) => *x,
                None => vk::PresentModeKHR::FIFO,
            };
        }
    }

    pub fn destroy(self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

pub struct DebugUtils {
    device: ext::debug_utils::Device,
}

impl DebugUtils {
    pub fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            device: ext::debug_utils::Device::new(&instance, &device),
        }
    }

    pub fn set_object_name<T: vk::Handle>(&self, handle: T, name: &str) {
        unsafe {
            let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(CStr::from_ptr(
                    (name.to_string() + "\0").as_ptr() as *const i8
                ));

            self.device.set_debug_utils_object_name(&name_info).unwrap();
        }
    }
}

pub struct Device {
    pub device: ash::Device,

    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
}

impl Device {
    pub fn new(instance: &ash::Instance, physical_device_data: &PhysicalDeviceData) -> Self {
        let req_exts = [
            vk::KHR_SWAPCHAIN_NAME.as_ptr() as *const i8,
            vk::KHR_RAY_TRACING_POSITION_FETCH_NAME.as_ptr() as *const i8,
            vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ptr() as *const i8,
            vk::KHR_RAY_TRACING_PIPELINE_NAME.as_ptr() as *const i8,
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ptr() as *const i8,
            vk::EXT_ROBUSTNESS2_NAME.as_ptr() as *const i8,
            vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr() as *const i8,
            vk::EXT_MESH_SHADER_NAME.as_ptr() as *const i8,
        ];

        let mut mesh_shader_feats = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .mesh_shader(true)
            .task_shader(true);
        let mut rt_pos_fetch_feats =
            vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
                .ray_tracing_position_fetch(true);
        let mut rt_pipe_feats =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut accel_struct_feats = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut dyn_rend_feats =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut sync2_feats =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut feats11 =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut feats12 = vk::PhysicalDeviceVulkan12Features::default()
            .uniform_and_storage_buffer8_bit_access(true)
            .shader_int8(true)
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .storage_push_constant8(true);
        let mut feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut mesh_shader_feats)
            .push_next(&mut rt_pos_fetch_feats)
            .push_next(&mut rt_pipe_feats)
            .push_next(&mut accel_struct_feats)
            .push_next(&mut dyn_rend_feats)
            .push_next(&mut sync2_feats)
            .push_next(&mut feats11)
            .push_next(&mut feats12);

        // Go through the retrieved queue family indices and create the device queue create infos.
        // If there are more than one queue family share the same index, then more than one queue is requested in the queue create info
        let mut d_q_cis: Vec<vk::DeviceQueueCreateInfo> = Vec::new();

        match d_q_cis.iter_mut().find(|d_q_ci| {
            d_q_ci.queue_family_index == physical_device_data.graphics_queue_family_index
        }) {
            Some(x) => {
                x.queue_count += 1;
            }
            None => {
                d_q_cis.push(vk::DeviceQueueCreateInfo {
                    queue_family_index: physical_device_data.graphics_queue_family_index,
                    queue_count: 1,
                    ..Default::default()
                });
            }
        }

        match d_q_cis.iter_mut().find(|d_q_ci| {
            d_q_ci.queue_family_index == physical_device_data.compute_queue_family_index
        }) {
            Some(x) => {
                x.queue_count += 1;
            }
            None => {
                d_q_cis.push(vk::DeviceQueueCreateInfo {
                    queue_family_index: physical_device_data.compute_queue_family_index,
                    queue_count: 1,
                    ..Default::default()
                });
            }
        }

        match d_q_cis.iter_mut().find(|d_q_ci| {
            d_q_ci.queue_family_index == physical_device_data.transfer_queue_family_index
        }) {
            Some(x) => {
                x.queue_count += 1;
            }
            None => {
                d_q_cis.push(vk::DeviceQueueCreateInfo {
                    queue_family_index: physical_device_data.transfer_queue_family_index,
                    queue_count: 1,
                    ..Default::default()
                });
            }
        }

        for d_q_ci in &mut d_q_cis {
            let priorities: Vec<f32> = vec![1f32; d_q_ci.queue_count as usize];

            d_q_ci.p_queue_priorities = priorities.as_ptr();
        }

        let create_info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&req_exts)
            .queue_create_infos(&d_q_cis)
            .push_next(&mut feats2);

        unsafe {
            let device = instance
                .create_device(physical_device_data.physical_device, &create_info, None)
                .unwrap();

            // Loop through the device queue create infos and retrieve the queues.
            let graphics_queue = match d_q_cis.iter().enumerate().find(|&d_q_ci| {
                d_q_ci.1.queue_family_index == physical_device_data.graphics_queue_family_index
            }) {
                Some(x) => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: physical_device_data.graphics_queue_family_index,
                    queue_index: x.1.queue_count - 1, // hardcoded since we need just one queue
                    ..Default::default()
                }),
                None => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: 0,
                    queue_index: 0,
                    ..Default::default()
                }),
            };

            let compute_queue = match d_q_cis.iter().enumerate().find(|&d_q_ci| {
                d_q_ci.1.queue_family_index == physical_device_data.compute_queue_family_index
            }) {
                Some(x) => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: physical_device_data.compute_queue_family_index,
                    queue_index: x.1.queue_count - 1, // hardcoded since we need just one queue
                    ..Default::default()
                }),
                None => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: 0,
                    queue_index: 0,
                    ..Default::default()
                }),
            };

            let transfer_queue = match d_q_cis.iter().enumerate().find(|&d_q_ci| {
                d_q_ci.1.queue_family_index == physical_device_data.transfer_queue_family_index
            }) {
                Some(x) => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: physical_device_data.transfer_queue_family_index,
                    queue_index: x.1.queue_count - 1, // hardcoded since we need just one queue
                    ..Default::default()
                }),
                None => device.get_device_queue2(&vk::DeviceQueueInfo2 {
                    queue_family_index: 0,
                    queue_index: 0,
                    ..Default::default()
                }),
            };

            Self {
                device,
                graphics_queue,
                compute_queue,
                transfer_queue,
            }
        }
    }

    pub fn destroy(self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub loader: khr::swapchain::Device,

    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub image_count: u32,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        surface: &SurfaceData,
        queue_family_indices: &[u32],
        debug_utils: &DebugUtils,
    ) -> Self {
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.surface)
            .min_image_count(surface.capabilities.surface_capabilities.min_image_count)
            .image_format(surface.surface_format.surface_format.format)
            .image_color_space(surface.surface_format.surface_format.color_space)
            .image_extent(surface.capabilities.surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .queue_family_indices(queue_family_indices)
            .pre_transform(surface.capabilities.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface.present_mode);

        let mut iv_ci = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface.surface_format.surface_format.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut image_views = Vec::new();

        let loader = ash::khr::swapchain::Device::new(instance, device);
        let swapchain;
        let mut images = vec![];

        unsafe {
            swapchain = loader.create_swapchain(&create_info, None).unwrap();
            images = loader.get_swapchain_images(swapchain).unwrap();

            for image in images.iter().as_ref() {
                iv_ci = iv_ci.image(*image);

                image_views.push(device.create_image_view(&iv_ci, None).unwrap());
            }

            if cfg!(debug_assertions) {
                debug_utils.set_object_name(swapchain, "swapchain");

                for (index, image) in images.iter().enumerate() {
                    let name = "swapchain image ".to_owned() + index.to_string().as_str();
                    debug_utils.set_object_name(*image, name.as_str());
                }

                for (index, image_view) in image_views.iter().enumerate() {
                    let name = "swapchain image view ".to_owned() + index.to_string().as_str();
                    debug_utils.set_object_name(*image_view, name.as_str());
                }
            }
        }

        Self {
            loader,
            swapchain,
            images,
            image_views,
            image_count: surface.capabilities.surface_capabilities.min_image_count,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);

            for image_view in &self.image_views {
                device.destroy_image_view(*image_view, None);
            }
        }
    }
}

/// Trait for VkMem or GpuAllocator, so any of them could be used by just creating the AllocatorType object instance accordingly.
/// VkMemAllocator::new() or GpuAllocatorAllocator::new()
pub trait AllocatorTrait {
    fn create_buffer_on_host_with_data<T>(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        data: &[T],
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource;

    fn create_buffer_on_host_with_size(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        size: vk::DeviceSize,
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource;

    fn create_buffer_on_device(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        data_size: vk::DeviceSize,
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource;

    fn create_image(
        &mut self,
        device: &ash::Device,
        extent: &vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        queue_family_indices: &[u32],
        debug_utils: &DebugUtils,
        name: &str,
    ) -> ImageResource;

    fn destroy_buffer_resource(&mut self, device: &ash::Device, buffer: BufferResource);
    fn destroy_image_resource(&mut self, device: &ash::Device, image: ImageResource);
    fn destroy_buffer_and_allocation(
        &mut self,
        device: &ash::Device,
        buffer: vk::Buffer,
        allocation: &mut AllocationType,
    );
    fn destroy(self);
}

#[derive(Default)]
pub enum AllocatorType {
    GpuAllocator(GpuAllocatorAllocator),
    VkMem(VkMemAllocator),
    #[default]
    None,
}

#[derive(Debug, Default)]
pub enum AllocationType {
    GpuAllocatorAllocation(gpu_allocator::vulkan::Allocation),
    VkMemAllocation(vk_mem::Allocation),
    #[default]
    None,
}

impl AllocatorTrait for AllocatorType {
    fn create_buffer_on_host_with_data<T>(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        data: &[T],
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource {
        unsafe {
            let mut buffer = vk::Buffer::null();
            let mut allocation_type = AllocationType::None;
            let mut mapped_ptr = None;
            let mut memory = vk::DeviceMemory::null();
            let mut size = 0;
            let mut offset = 0;
            let mut data_size = 0;
            let mut device_address = 0;
            let mut device_or_host_address = vk::DeviceOrHostAddressKHR::default();
            let mut device_or_host_address_const = vk::DeviceOrHostAddressConstKHR::default();
            let alignment = min_alignment.unwrap_or(0);

            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    buffer = device
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size((data.len() * size_of::<T>()) as vk::DeviceSize)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            None,
                        )
                        .unwrap();

                    let mut requirements = vk::MemoryRequirements2::default();
                    device.get_buffer_memory_requirements2(
                        &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                        &mut requirements,
                    );

                    requirements.memory_requirements = requirements
                        .memory_requirements
                        .alignment(requirements.memory_requirements.alignment.max(alignment));

                    let allocation = allocator
                        .allocator
                        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                            name: &(name.to_owned() + " allocation"),
                            requirements: requirements.memory_requirements,
                            location: gpu_allocator::MemoryLocation::CpuToGpu,
                            linear: true,
                            allocation_scheme:
                                gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                        })
                        .unwrap();

                    let bind_infos = [vk::BindBufferMemoryInfo::default()
                        .buffer(buffer)
                        .memory(allocation.memory())
                        .memory_offset(allocation.offset())];

                    device.bind_buffer_memory2(&bind_infos).unwrap();

                    std::ptr::copy(
                        data.as_ptr(),
                        allocation.mapped_ptr().unwrap().as_ptr() as *mut T,
                        data.len(),
                    );

                    mapped_ptr = Some(allocation.mapped_ptr().unwrap().as_ptr());
                    memory = allocation.memory();
                    size = allocation.size();
                    offset = allocation.offset();
                    data_size = (data.len() * size_of::<T>()) as vk::DeviceSize;
                    allocation_type = AllocationType::GpuAllocatorAllocation(allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }
                }
                AllocatorType::VkMem(allocator) => {
                    let (this_buffer, allocation) = allocator
                        .allocator
                        .create_buffer_with_alignment(
                            &vk::BufferCreateInfo::default()
                                .size((data.len() * size_of::<T>()) as vk::DeviceSize)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::Auto,
                                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                    | vk_mem::AllocationCreateFlags::MAPPED,
                                preferred_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
                                ..Default::default()
                            },
                            alignment,
                        )
                        .unwrap();

                    let allocation_info = allocator.allocator.get_allocation_info2(&allocation);

                    std::ptr::copy(
                        data.as_ptr(),
                        allocation_info.allocation_info.mapped_data as *mut T,
                        data.len(),
                    );

                    mapped_ptr = Some(allocation_info.allocation_info.mapped_data);
                    memory = allocation_info.allocation_info.device_memory;
                    size = allocation_info.allocation_info.size;
                    offset = allocation_info.allocation_info.offset;
                    data_size = (data.len() * size_of::<T>()) as vk::DeviceSize;
                    allocation_type = AllocationType::VkMemAllocation(allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(this_buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }

                    buffer = this_buffer;
                }
                _ => {}
            }

            device_address = device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            device_or_host_address.device_address = device_address;
            device_or_host_address_const.device_address = device_address;

            BufferResource {
                descriptor_info: vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE),
                mapped_ptr,
                memory,
                size,
                offset,
                data_size,
                device_address,
                device_or_host_address,
                device_or_host_address_const,
                allocation_type,
            }
        }
    }

    fn create_buffer_on_host_with_size(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        data_size: vk::DeviceSize,
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource {
        unsafe {
            let mut buffer = vk::Buffer::null();
            let mut allocation_type = AllocationType::None;
            let mut mapped_ptr = None;
            let mut memory = vk::DeviceMemory::null();
            let mut size = 0;
            let mut offset = 0;
            let mut device_address = 0;
            let mut device_or_host_address = vk::DeviceOrHostAddressKHR::default();
            let mut device_or_host_address_const = vk::DeviceOrHostAddressConstKHR::default();
            let alignment = min_alignment.unwrap_or(0);

            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    buffer = device
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size(data_size)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            None,
                        )
                        .unwrap();

                    let mut requirements = vk::MemoryRequirements2::default();
                    device.get_buffer_memory_requirements2(
                        &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                        &mut requirements,
                    );

                    requirements.memory_requirements = requirements
                        .memory_requirements
                        .alignment(requirements.memory_requirements.alignment.max(alignment));

                    let allocation = allocator
                        .allocator
                        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                            name: &(name.to_owned() + " allocation"),
                            requirements: requirements.memory_requirements,
                            location: gpu_allocator::MemoryLocation::CpuToGpu,
                            linear: true,
                            allocation_scheme:
                                gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                        })
                        .unwrap();

                    let bind_infos = [vk::BindBufferMemoryInfo::default()
                        .buffer(buffer)
                        .memory(allocation.memory())
                        .memory_offset(allocation.offset())];

                    device.bind_buffer_memory2(&bind_infos).unwrap();

                    mapped_ptr = Some(allocation.mapped_ptr().unwrap().as_ptr());
                    memory = allocation.memory();
                    size = allocation.size();
                    offset = allocation.offset();
                    allocation_type = AllocationType::GpuAllocatorAllocation(allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }
                }
                AllocatorType::VkMem(allocator) => {
                    let (this_buffer, allocation) = allocator
                        .allocator
                        .create_buffer_with_alignment(
                            &vk::BufferCreateInfo::default()
                                .size(data_size)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::Auto,
                                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                                    | vk_mem::AllocationCreateFlags::MAPPED,
                                preferred_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
                                ..Default::default()
                            },
                            alignment,
                        )
                        .unwrap();

                    let allocation_info = allocator.allocator.get_allocation_info2(&allocation);

                    mapped_ptr = Some(allocation_info.allocation_info.mapped_data);
                    memory = allocation_info.allocation_info.device_memory;
                    size = allocation_info.allocation_info.size;
                    offset = allocation_info.allocation_info.offset;

                    allocation_type = AllocationType::VkMemAllocation(allocation);
                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(this_buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }

                    buffer = this_buffer;
                }
                _ => {}
            }

            device_address = device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            device_or_host_address.device_address = device_address;
            device_or_host_address_const.device_address = device_address;

            BufferResource {
                descriptor_info: vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE),
                mapped_ptr,
                memory,
                size,
                offset,
                data_size,
                device_address,
                device_or_host_address,
                device_or_host_address_const,
                allocation_type,
            }
        }
    }

    fn create_buffer_on_device(
        &mut self,
        device: &ash::Device,
        usage: vk::BufferUsageFlags,
        data_size: vk::DeviceSize,
        debug_utils: &DebugUtils,
        min_alignment: Option<vk::DeviceSize>,
        name: &str,
    ) -> BufferResource {
        unsafe {
            let mut buffer = vk::Buffer::null();
            let mut allocation_type = AllocationType::None;
            let mapped_ptr = None;
            let mut memory = vk::DeviceMemory::null();
            let mut size = 0;
            let mut offset = 0;
            let mut device_address = 0;
            let mut device_or_host_address = vk::DeviceOrHostAddressKHR::default();
            let mut device_or_host_address_const = vk::DeviceOrHostAddressConstKHR::default();
            let alignment = min_alignment.unwrap_or(0);

            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    buffer = device
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size(data_size)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            None,
                        )
                        .unwrap();

                    let mut requirements = vk::MemoryRequirements2::default();
                    device.get_buffer_memory_requirements2(
                        &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                        &mut requirements,
                    );

                    requirements.memory_requirements.alignment =
                        requirements.memory_requirements.alignment.max(alignment);

                    let allocation = allocator
                        .allocator
                        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                            name: &(name.to_owned() + " allocation"),
                            requirements: requirements.memory_requirements,
                            location: gpu_allocator::MemoryLocation::GpuOnly,
                            linear: true,
                            allocation_scheme:
                                gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                        })
                        .unwrap();

                    let bind_infos = [vk::BindBufferMemoryInfo::default()
                        .buffer(buffer)
                        .memory(allocation.memory())
                        .memory_offset(allocation.offset())];

                    device.bind_buffer_memory2(&bind_infos).unwrap();

                    memory = allocation.memory();
                    size = allocation.size();
                    offset = allocation.offset();
                    allocation_type = AllocationType::GpuAllocatorAllocation(allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }
                }
                AllocatorType::VkMem(allocator) => {
                    let (this_buffer, allocation) = allocator
                        .allocator
                        .create_buffer_with_alignment(
                            &vk::BufferCreateInfo::default()
                                .size(data_size)
                                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::Auto,
                                ..Default::default()
                            },
                            alignment,
                        )
                        .unwrap();

                    let allocation_info = allocator.allocator.get_allocation_info2(&allocation);

                    memory = allocation_info.allocation_info.device_memory;
                    size = allocation_info.allocation_info.size;
                    offset = allocation_info.allocation_info.offset;

                    allocation_type = AllocationType::VkMemAllocation(allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(this_buffer, &(name.to_owned() + " buffer"));
                        debug_utils.set_object_name(memory, &(name.to_owned() + " memory"));
                    }

                    buffer = this_buffer;
                }
                AllocatorType::None => {}
            }

            device_address = device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            device_or_host_address.device_address = device_address;
            device_or_host_address_const.device_address = device_address;

            BufferResource {
                descriptor_info: vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE),
                mapped_ptr,
                memory,
                size,
                offset,
                data_size,
                device_address,
                device_or_host_address,
                device_or_host_address_const,
                allocation_type,
            }
        }
    }

    fn create_image(
        &mut self,
        device: &ash::Device,
        extent: &vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        queue_family_indices: &[u32],
        debug_utils: &DebugUtils,
        name: &str,
    ) -> ImageResource {
        let mut image = vk::Image::null();
        let mut image_view = vk::ImageView::null();
        let mut sampler = vk::Sampler::null();
        let mut allocation_type = AllocationType::None;

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(
                vk::Extent3D::default()
                    .width(extent.width)
                    .height(extent.height)
                    .depth(1),
            )
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(queue_family_indices);

        let aspect_mask = match format {
            vk::Format::D32_SFLOAT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D16_UNORM
            | vk::Format::D16_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        };

        let mut image_view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .level_count(1)
                    .layer_count(1),
            );

        let sampler_create_info = vk::SamplerCreateInfo::default();

        unsafe {
            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    image = device.create_image(&image_create_info, None).unwrap();

                    let mut requirements = vk::MemoryRequirements2::default();
                    device.get_image_memory_requirements2(
                        &vk::ImageMemoryRequirementsInfo2::default().image(image),
                        &mut requirements,
                    );

                    let allocation = allocator
                        .allocator
                        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                            name: &(name.to_owned() + " allocation"),
                            requirements: requirements.memory_requirements,
                            location: gpu_allocator::MemoryLocation::GpuOnly,
                            linear: false,
                            allocation_scheme:
                                gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                        })
                        .unwrap();

                    let bind_infos = [vk::BindImageMemoryInfo::default()
                        .image(image)
                        .memory(allocation.memory())
                        .memory_offset(allocation.offset())];

                    device.bind_image_memory2(&bind_infos).unwrap();

                    image_view_create_info = image_view_create_info.image(image);

                    sampler = device.create_sampler(&sampler_create_info, None).unwrap();
                    image_view = device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap();

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(image, &(name.to_owned() + " image"));
                        debug_utils.set_object_name(sampler, &(name.to_owned() + " sampler"));
                        debug_utils.set_object_name(image_view, &(name.to_owned() + " image view"));
                        debug_utils
                            .set_object_name(allocation.memory(), &(name.to_owned() + " memory"));
                    }
                    allocation_type = AllocationType::GpuAllocatorAllocation(allocation);
                }
                AllocatorType::VkMem(allocator) => {
                    let allocation;
                    (image, allocation) = allocator
                        .allocator
                        .create_image(
                            &image_create_info,
                            &vk_mem::AllocationCreateInfo {
                                usage: vk_mem::MemoryUsage::Auto,
                                ..Default::default()
                            },
                        )
                        .unwrap();

                    image_view_create_info = image_view_create_info.image(image);

                    sampler = device.create_sampler(&sampler_create_info, None).unwrap();
                    image_view = device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap();

                    let allocation_info = allocator.allocator.get_allocation_info2(&allocation);

                    if cfg!(debug_assertions) {
                        debug_utils.set_object_name(image, &(name.to_owned() + " image"));
                        debug_utils.set_object_name(sampler, &(name.to_owned() + " sampler"));
                        debug_utils.set_object_name(image_view, &(name.to_owned() + " image view"));
                        debug_utils.set_object_name(
                            allocation_info.allocation_info.device_memory,
                            &(name.to_owned() + " memory"),
                        );
                    }
                    allocation_type = AllocationType::VkMemAllocation(allocation);
                }
                _ => {}
            }
        }

        ImageResource {
            descriptor_info: vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(image_view),
            image,
            allocation_type,
            width: extent.width,
            height: extent.height,
        }
    }

    fn destroy_buffer_resource(&mut self, device: &ash::Device, buffer: BufferResource) {
        unsafe {
            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    device.destroy_buffer(buffer.descriptor_info.buffer, None);
                    match buffer.allocation_type {
                        AllocationType::GpuAllocatorAllocation(allocation) => {
                            allocator.allocator.free(allocation).unwrap();
                        }
                        _ => {}
                    }
                }
                AllocatorType::VkMem(allocator) => match buffer.allocation_type {
                    AllocationType::VkMemAllocation(mut allocation) => {
                        allocator
                            .allocator
                            .destroy_buffer(buffer.descriptor_info.buffer, &mut allocation);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    fn destroy_buffer_and_allocation(
        &mut self,
        device: &ash::Device,
        buffer: vk::Buffer,
        allocation: &mut AllocationType,
    ) {
        unsafe {
            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    device.destroy_buffer(buffer, None);
                    if let AllocationType::GpuAllocatorAllocation(allocation) = allocation {
                        allocator
                            .allocator
                            .free(std::mem::take(allocation))
                            .unwrap();
                    }
                }
                AllocatorType::VkMem(allocator) => {
                    if let AllocationType::VkMemAllocation(allocation) = allocation {
                        allocator.allocator.destroy_buffer(buffer, allocation);
                    }
                }
                AllocatorType::None => {}
            }
        }
    }

    fn destroy_image_resource(&mut self, device: &ash::Device, image: ImageResource) {
        unsafe {
            match self {
                AllocatorType::GpuAllocator(allocator) => {
                    device.destroy_image(image.image, None);
                    device.destroy_image_view(image.descriptor_info.image_view, None);
                    device.destroy_sampler(image.descriptor_info.sampler, None);

                    match image.allocation_type {
                        AllocationType::GpuAllocatorAllocation(allocation) => {
                            allocator.allocator.free(allocation).unwrap();
                        }
                        _ => {}
                    }
                }
                AllocatorType::VkMem(allocator) => {
                    device.destroy_image_view(image.descriptor_info.image_view, None);
                    device.destroy_sampler(image.descriptor_info.sampler, None);

                    match image.allocation_type {
                        AllocationType::VkMemAllocation(mut allocation) => {
                            allocator
                                .allocator
                                .destroy_image(image.image, &mut allocation);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    fn destroy(self) {
        match self {
            AllocatorType::GpuAllocator(allocator) => std::mem::drop(allocator.allocator),
            AllocatorType::VkMem(allocator) => std::mem::drop(allocator.allocator),
            _ => {}
        }
    }
}

pub struct GpuAllocatorAllocator {
    allocator: gpu_allocator::vulkan::Allocator,
}

impl GpuAllocatorAllocator {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: &vk::PhysicalDevice,
    ) -> Self {
        let allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: *physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: true,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .unwrap();

        Self { allocator }
    }
}

pub struct VkMemAllocator {
    allocator: vk_mem::Allocator,
}

impl VkMemAllocator {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: &vk::PhysicalDevice,
        vk_api_version: u32,
    ) -> Self {
        unsafe {
            let mut create_info =
                vk_mem::AllocatorCreateInfo::new(instance, device, *physical_device);
            create_info.vulkan_api_version = vk_api_version;
            create_info.flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

            let allocator = vk_mem::Allocator::new(create_info).unwrap();

            Self { allocator }
        }
    }
}

pub struct BufferResource {
    pub descriptor_info: vk::DescriptorBufferInfo,
    pub mapped_ptr: Option<*mut c_void>,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub offset: vk::DeviceSize,
    pub data_size: vk::DeviceSize,
    pub device_address: vk::DeviceAddress,
    pub device_or_host_address: vk::DeviceOrHostAddressKHR,
    pub device_or_host_address_const: vk::DeviceOrHostAddressConstKHR,
    pub allocation_type: AllocationType,
}

impl BufferResource {
    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.descriptor_info.buffer
    }
}

// ONLY to facilitate the destruction of the resource, since the GPUAllocator moves the "Allocation" object to free it, and std::mem::take is used to "take" the resource from a mut reference of the parent.
// This should immediately followed by creation of the new resource
impl Default for BufferResource {
    fn default() -> Self {
        Self {
            descriptor_info: vk::DescriptorBufferInfo::default(),
            mapped_ptr: None,
            memory: vk::DeviceMemory::null(),
            size: 0,
            offset: 0,
            data_size: 0,
            device_address: 0,
            device_or_host_address: vk::DeviceOrHostAddressKHR::default(),
            device_or_host_address_const: vk::DeviceOrHostAddressConstKHR::default(),
            allocation_type: AllocationType::None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ImageResource {
    pub descriptor_info: vk::DescriptorImageInfo,
    pub image: vk::Image,
    pub allocation_type: AllocationType,

    pub width: u32,
    pub height: u32,
}

/// One place to batch all the transfers for given situation. 
/// Change image layouts, copy buffer to buffer, to image etc etc.
pub struct DataHelper {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub queue: vk::Queue,
    pub semaphore: vk::Semaphore,
    pub semaphore_value: u64,
    pub queue_family_index: u32,
}

impl DataHelper {
    pub fn new(
        device: &ash::Device,
        queue: &vk::Queue,
        queue_family_index: u32,
        debug_utils: &DebugUtils,
    ) -> Self {
        unsafe {
            {
                let command_pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                            .queue_family_index(queue_family_index),
                        None,
                    )
                    .unwrap();

                let mut sem_tl_ci = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE);
                let semaphore = device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut sem_tl_ci),
                        None,
                    )
                    .unwrap();
                let command_buffer = device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0];

                if cfg!(debug_assertions) {
                    debug_utils.set_object_name(command_pool, "data helpers command pool");
                    debug_utils.set_object_name(command_buffer, "data helpers command buffer");
                    debug_utils.set_object_name(semaphore, "data helpers semaphore");
                }

                Self {
                    command_pool,
                    queue: *queue,
                    command_buffer,
                    semaphore,
                    semaphore_value: 0,
                    queue_family_index,
                }
            }
        }
    }

    pub fn record_batch(&self, device: &ash::Device) {
        let semaphores = [self.semaphore];
        let values = [self.semaphore_value];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.wait_semaphores(&wait_info, u64::MAX).unwrap();
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .unwrap();
        }
    }

    pub fn clear_image(&self, device: &ash::Device, image: &vk::Image) {
        let clear_color = vk::ClearColorValue {
            float32: [0f32, 0f32, 0f32, 1f32],
        };

        unsafe {
            device.cmd_clear_color_image(
                self.command_buffer,
                *image,
                vk::ImageLayout::GENERAL,
                &clear_color,
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1)],
            );
        }
    }

    pub fn change_image_layout(
        &self,
        device: &ash::Device,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        aspect_mask: vk::ImageAspectFlags,
        image: &vk::Image,
    ) {
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .level_count(1)
                    .layer_count(1)
                    .aspect_mask(aspect_mask),
            )
            .image(*image)];

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);

        unsafe {
            device.cmd_pipeline_barrier2(self.command_buffer, &dependency_info);
        }
    }

    pub fn copy_buffer_to_buffer(
        &self,
        device: &ash::Device,
        src_buffer: &vk::Buffer,
        dst_buffer: &vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let regions = [vk::BufferCopy2::default().size(size)];
        let copy_buffer_info = vk::CopyBufferInfo2::default()
            .src_buffer(*src_buffer)
            .dst_buffer(*dst_buffer)
            .regions(&regions);

        unsafe {
            device.cmd_copy_buffer2(self.command_buffer, &copy_buffer_info);
        }
    }

    pub fn copy_buffer_to_image(
        &self,
        device: &ash::Device,
        src_buffer: &vk::Buffer,
        dst_image: &vk::Image,
        extent: vk::Extent3D,
        offset: Option<vk::Offset3D>,
    ) {
        let regions = [vk::BufferImageCopy2::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .layer_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            )
            .image_extent(extent)
            .image_offset(offset.unwrap_or_default())];

        let copy_buff_info = vk::CopyBufferToImageInfo2::default()
            .src_buffer(*src_buffer)
            .dst_image(*dst_image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);

        unsafe {
            device.cmd_copy_buffer_to_image2(self.command_buffer, &copy_buff_info);
        }
    }

    pub fn insert_memory_barrier(
        &self,
        device: &ash::Device,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) {
        let mem_bars = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)];
        let dep_info = vk::DependencyInfo::default().memory_barriers(&mem_bars);

        unsafe {
            device.cmd_pipeline_barrier2(self.command_buffer, &dep_info);
        }
    }

    pub fn build_acceleration_structure(
        &self,
        device: &ash::khr::acceleration_structure::Device,
        build_geom_infos: &[vk::AccelerationStructureBuildGeometryInfoKHR],
        build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]],
    ) {
        unsafe {
            device.cmd_build_acceleration_structures(
                self.command_buffer,
                build_geom_infos,
                build_range_infos,
            );
        }
    }

    pub fn submit_batch(
        &mut self,
        device: &ash::Device,
        wait_semaphore: Option<vk::Semaphore>,
        wait_semaphore_value: Option<u64>,
        wait_stage_mask: Option<vk::PipelineStageFlags2>,
    ) {
        let mut wait_sem_infos = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.semaphore)
                .value(self.semaphore_value)
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
        ];

        if let Some(wait_semaphore) = wait_semaphore {
            wait_sem_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(wait_semaphore)
                    .value(wait_semaphore_value.unwrap_or(0))
                    .stage_mask(wait_stage_mask.unwrap_or(vk::PipelineStageFlags2::TOP_OF_PIPE)),
            );
        }

        let cmd_buff_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
        let sig_sem_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.semaphore_value + 1)
            .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)];

        let submit_infos = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_sem_infos)
            .command_buffer_infos(&cmd_buff_infos)
            .signal_semaphore_infos(&sig_sem_infos)];

        unsafe {
            device.end_command_buffer(self.command_buffer).unwrap();
            device
                .queue_submit2(self.queue, &submit_infos, vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(self.queue).unwrap();
        }

        self.semaphore_value += 1;
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_semaphore(self.semaphore, None);
        }
    }
}

/// Make handling of vulkan render "frames" convenient, with semaphores, command buffers for each frame in flight.
pub struct FrameObjects {
    pub semaphores: Vec<vk::Semaphore>,
    pub semaphore_values: Vec<u64>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_pool: vk::CommandPool,
    pub max_frames_in_flight: usize,
    pub frame_in_flight: usize,
}

impl FrameObjects {
    pub fn new(
        device: &ash::Device,
        queue_family_index: u32,
        max_frames_in_flight: usize,
        debug_utils: &DebugUtils,
    ) -> Self {
        let semaphore_values = vec![1u64; max_frames_in_flight];
        let mut semaphores = Vec::new();

        let mut sem_type_ci =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type_ci);
        let mut sem_sig_info = vk::SemaphoreSignalInfo::default().value(1);
        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let command_pool;
        let command_buffers;

        unsafe {
            command_pool = device.create_command_pool(&cmd_pool_ci, None).unwrap();
            let cmd_buff_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(max_frames_in_flight as u32);
            command_buffers = device.allocate_command_buffers(&cmd_buff_ai).unwrap();

            for _ in 0..max_frames_in_flight {
                let semaphore = device.create_semaphore(&sem_ci, None).unwrap();
                sem_sig_info = sem_sig_info.semaphore(semaphore);
                device.signal_semaphore(&sem_sig_info).unwrap();
                semaphores.push(semaphore);
            }
        }

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(command_pool, "frame objects command pool");

            for (index, semaphore) in semaphores.iter().enumerate() {
                let name = "frame objects semaphore ".to_owned() + index.to_string().as_str();
                debug_utils.set_object_name(*semaphore, name.as_str());
            }

            for (index, command_buffer) in command_buffers.iter().enumerate() {
                let name = "frame objects command buffer ".to_owned() + index.to_string().as_str();
                debug_utils.set_object_name(*command_buffer, name.as_str());
            }
        }

        Self {
            semaphores,
            semaphore_values,
            max_frames_in_flight,
            frame_in_flight: 0,
            command_pool,
            command_buffers,
        }
    }

    pub fn get_command_buffer(&self) -> &vk::CommandBuffer {
        &self.command_buffers[self.frame_in_flight as usize]
    }

    pub fn get_semaphore(&self) -> &vk::Semaphore {
        &self.semaphores[self.frame_in_flight as usize]
    }

    pub fn get_frame_sem_value(&self) -> u64 {
        self.semaphore_values[self.frame_in_flight as usize]
    }

    pub fn get_frame_in_flight(&self) -> usize {
        self.frame_in_flight
    }

    /// Go to the next frame in flight. Increment the semaphore value to wait on at the start of the next frame.
    pub fn next_frame(&mut self) {
        self.semaphore_values[self.frame_in_flight as usize] += 1;
        self.frame_in_flight = (self.frame_in_flight + 1) % self.max_frames_in_flight;
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);

            for s in 0..self.max_frames_in_flight as usize {
                device.destroy_semaphore(self.semaphores[s], None);
            }
        }
    }
}

struct DataConverterPipelineData {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DataConverterPipelineDataPushConstants {
    input_r8: vk::DeviceAddress,
    input_rg8: vk::DeviceAddress,
    input_rgb8: vk::DeviceAddress,
    input_rgba8: vk::DeviceAddress,

    image_width: u32,
    image_height: u32,
    image_channels: u32,

    dst_image_index: u32,
    srgb_to_linear: bool,
}

unsafe impl bytemuck::NoUninit for DataConverterPipelineDataPushConstants {}

/// Pipeline for the DataConverter compute shader
impl DataConverterPipelineData {
    pub fn new(device: &ash::Device, debug_utils: &DebugUtils, textures_count: u32) -> Self {
        let spirv_data = std::fs::read(
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
                + ("/shaders/data_conversion.slang.spv"),
        )
        .expect("Could not read data_conversion.slang.spv");

        let shader_mod;
        unsafe {
            shader_mod = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(spirv_data.align_to::<u32>().1),
                    None,
                )
                .unwrap();

            if cfg!(debug_assertions) {
                debug_utils.set_object_name(shader_mod, "data_conversion shader");
            }
        }

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_mod)
            .name(CStr::from_bytes_with_nul("main\0".as_bytes()).unwrap());

        let dsl_0_binds = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(textures_count)];

        let binding_flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND];
        let mut dsl_0_bind_flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let dsl_cis = [vk::DescriptorSetLayoutCreateInfo::default()
            .push_next(&mut dsl_0_bind_flags_ci)
            .bindings(&dsl_0_binds)];

        let mut descriptor_set_layouts = vec![];
        unsafe {
            for dsl_ci in dsl_cis {
                descriptor_set_layouts
                    .push(device.create_descriptor_set_layout(&dsl_ci, None).unwrap());
            }
        }

        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<DataConverterPipelineDataPushConstants>() as u32)];

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&pc_ranges)
            .set_layouts(&descriptor_set_layouts);

        let layout;
        unsafe {
            layout = device.create_pipeline_layout(&layout_ci, None).unwrap();
        }

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(stage);

        let pipeline;
        unsafe {
            pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .unwrap()[0];
        }

        if cfg!(debug_assertions) {
            for dsl in &descriptor_set_layouts {
                debug_utils.set_object_name(*dsl, "data converter descriptor set layout");
            }

            debug_utils.set_object_name(pipeline, "data converter pipeline");
            debug_utils.set_object_name(layout, "data converter pipeline layout");
        }

        unsafe {
            device.destroy_shader_module(shader_mod, None);
        }

        Self {
            pipeline,
            layout,
            descriptor_set_layouts,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);

            device.destroy_pipeline(self.pipeline, None);

            for layout in &self.descriptor_set_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
}

/// Helps to convert images from n-channel to 4-channel RGBA, also converts srgb to linear.
/// Takes in a buffer and writes into a STORAGE image.
/// batch all the conversions and submit
pub struct DataConverter {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub queue: vk::Queue,
    pub semaphore: vk::Semaphore,
    pub semaphore_value: u64,
    pub queue_family_index: u32,

    pipeline_data: DataConverterPipelineData,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl DataConverter {
    pub fn new(
        device: &ash::Device,
        queue: &vk::Queue,
        queue_family_index: u32,
        debug_utils: &DebugUtils,
        image_descs: &[vk::DescriptorImageInfo],
    ) -> Self {
        unsafe {
            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family_index),
                    None,
                )
                .unwrap();

            let mut sem_tl_ci =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut sem_tl_ci),
                    None,
                )
                .unwrap();
            let command_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0];

            let pipeline_data =
                DataConverterPipelineData::new(device, debug_utils, image_descs.len() as u32);

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .descriptor_count(image_descs.len() as u32)
                .ty(vk::DescriptorType::STORAGE_IMAGE)];

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .unwrap();

            let ds_counts = [image_descs.len() as u32];
            let mut ds_vdcai = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&ds_counts);

            let ds_ai = vk::DescriptorSetAllocateInfo::default()
                .push_next(&mut ds_vdcai)
                .descriptor_pool(descriptor_pool)
                .set_layouts(&pipeline_data.descriptor_set_layouts);

            let descriptor_set = device.allocate_descriptor_sets(&ds_ai).unwrap()[0];

            let write_desc_set = [vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .descriptor_count(image_descs.len() as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_descs)];

            device.update_descriptor_sets(&write_desc_set, &[]);

            if cfg!(debug_assertions) {
                debug_utils.set_object_name(command_pool, "data converter command pool");
                debug_utils.set_object_name(command_buffer, "data converter command buffer");
                debug_utils.set_object_name(semaphore, "data converter semaphore");
                debug_utils.set_object_name(descriptor_pool, "data converter descriptor pool");
                debug_utils.set_object_name(descriptor_set, "data converter descriptor set");
            }

            Self {
                command_pool,
                queue: *queue,
                command_buffer,
                semaphore,
                semaphore_value: 0,
                queue_family_index,
                pipeline_data,
                descriptor_pool,
                descriptor_set,
            }
        }
    }

    pub fn record_batch(&self, device: &ash::Device) {
        let semaphores = [self.semaphore];
        let values = [self.semaphore_value];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.wait_semaphores(&wait_info, u64::MAX).unwrap();
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .unwrap();
        }
    }

    pub fn change_image_layout(
        &self,
        device: &ash::Device,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        aspect_mask: vk::ImageAspectFlags,
        image: &vk::Image,
    ) {
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .level_count(1)
                    .layer_count(1)
                    .aspect_mask(aspect_mask),
            )
            .image(*image)];

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);

        unsafe {
            device.cmd_pipeline_barrier2(self.command_buffer, &dependency_info);
        }
    }

    pub fn convert(
        &self,
        device: &ash::Device,
        src_buffer_addr: vk::DeviceAddress,
        dst_image_index: u32,
        image_data: (gltf::image::Format, u32, u32, bool),
    ) {
        unsafe {
            let mut pc = DataConverterPipelineDataPushConstants {
                image_width: image_data.1,
                image_height: image_data.2,
                dst_image_index,
                srgb_to_linear: image_data.3,
                ..Default::default()
            };

            match image_data.0 {
                gltf::image::Format::R8 => {
                    pc.image_channels = 1;
                    pc.input_r8 = src_buffer_addr;
                }
                gltf::image::Format::R8G8 => {
                    pc.image_channels = 2;
                    pc.input_rg8 = src_buffer_addr;
                }
                gltf::image::Format::R8G8B8 => {
                    pc.image_channels = 3;
                    pc.input_rgb8 = src_buffer_addr;
                }
                gltf::image::Format::R8G8B8A8 => {
                    pc.image_channels = 4;
                    pc.input_rgba8 = src_buffer_addr;
                }
                _ => {}
            }

            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_data.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_data.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_data.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );

            device.cmd_dispatch(
                self.command_buffer,
                image_data.1 / 32 + 1,
                image_data.2 / 32 + 1,
                1,
            );
        }
    }

    pub fn submit_batch(
        &mut self,
        device: &ash::Device,
        wait_semaphore: Option<vk::Semaphore>,
        wait_semaphore_value: Option<u64>,
        wait_stage_mask: Option<vk::PipelineStageFlags2>,
    ) {
        let mut wait_sem_infos = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.semaphore)
                .value(self.semaphore_value)
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
        ];

        if let Some(wait_semaphore) = wait_semaphore {
            wait_sem_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(wait_semaphore)
                    .value(wait_semaphore_value.unwrap_or(0))
                    .stage_mask(wait_stage_mask.unwrap_or(vk::PipelineStageFlags2::TOP_OF_PIPE)),
            );
        }

        let cmd_buff_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
        let sig_sem_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.semaphore_value + 1)
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];

        let submit_infos = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_sem_infos)
            .command_buffer_infos(&cmd_buff_infos)
            .signal_semaphore_infos(&sig_sem_infos)];

        unsafe {
            device.end_command_buffer(self.command_buffer).unwrap();
            device
                .queue_submit2(self.queue, &submit_infos, vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(self.queue).unwrap();
        }

        self.semaphore_value += 1;
    }

    pub fn destroy(self, device: &ash::Device) {
        self.pipeline_data.destroy(device);

        unsafe {
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_semaphore(self.semaphore, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

#[derive(Default)]
pub struct BLAS {
    pub accel_struct: vk::AccelerationStructureKHR,
    pub accel_struct_buffer: vk::Buffer,
    pub accel_struct_allocation: vulkan_objects::AllocationType,
}

use nalgebra_glm as glm;

impl BLAS {
    /// Build from triangle mesh
    pub fn new_from_mesh(
        device: &ash::Device,
        allocator: &mut AllocatorType,
        accel_struct_device: &ash::khr::acceleration_structure::Device,
        positions_addr: vk::DeviceOrHostAddressConstKHR,
        index_addr: vk::DeviceOrHostAddressConstKHR,
        vertex_count: u32,
        index_count: u32,
        scratch_buffer_alignment: vk::DeviceSize,
        compute_helper: &mut DataHelper,
        debug_utils: &DebugUtils,
        name: &str,
    ) -> Self {
        let mut accel_geom_data = vk::AccelerationStructureGeometryDataKHR::default();

        accel_geom_data.triangles = unsafe {
            accel_geom_data
                .triangles
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(positions_addr)
                .vertex_stride(size_of::<glm::Vec3>() as vk::DeviceSize)
                .max_vertex(vertex_count - 1)
                .index_type(vk::IndexType::UINT32)
                .index_data(index_addr)
        };

        accel_geom_data.triangles.s_type =
            vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

        let accel_geoms = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(accel_geom_data)];

        let mut accel_build_geom_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .geometries(&accel_geoms)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            accel_struct_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &accel_build_geom_info,
                &[index_count / 3],
                &mut size_info,
            );
        }

        let accel_struct_buffer_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            size_info.acceleration_structure_size,
            debug_utils,
            None,
            "BLAS",
        );

        let scratch_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            aligned_size!(size_info.build_scratch_size, scratch_buffer_alignment),
            debug_utils,
            Some(scratch_buffer_alignment),
            "BLAS scratch",
        );

        let accel_struct;
        unsafe {
            accel_struct = accel_struct_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(*accel_struct_buffer_resource.vk_buffer())
                        .size(size_info.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
                .unwrap();
        }

        let build_range_info =
            &[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(index_count / 3)];

        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] =
            &[build_range_info];

        accel_build_geom_info.dst_acceleration_structure = accel_struct;
        accel_build_geom_info.scratch_data = scratch_buffer.device_or_host_address;

        compute_helper.record_batch(device);
        compute_helper.build_acceleration_structure(
            accel_struct_device,
            &[accel_build_geom_info],
            &build_range_infos,
        );
        compute_helper.submit_batch(device, None, None, None);

        allocator.destroy_buffer_resource(device, scratch_buffer);

        let accel_struct_buffer = *accel_struct_buffer_resource.vk_buffer();
        let accel_struct_allocation = accel_struct_buffer_resource.allocation_type;

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(accel_struct, &(name.to_owned() + " BLAS"));
        }

        Self {
            accel_struct,
            accel_struct_allocation,
            accel_struct_buffer,
        }
    }

    /// Build a AABB data
    pub fn new_from_proc(
        device: &ash::Device,
        allocator: &mut AllocatorType,
        accel_struct_device: &ash::khr::acceleration_structure::Device,
        bbox_addr: vk::DeviceOrHostAddressConstKHR,
        scratch_buffer_alignment: vk::DeviceSize,
        compute_helper: &mut DataHelper,
        debug_utils: &DebugUtils,
        name: &str,
    ) -> Self {
        let mut accel_geom_data = vk::AccelerationStructureGeometryDataKHR::default();

        accel_geom_data.aabbs = unsafe {
            accel_geom_data
                .aabbs
                .data(bbox_addr)
                .stride(size_of::<vk::AabbPositionsKHR>() as vk::DeviceSize)
        };

        accel_geom_data.aabbs.s_type =
            vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;

        let accel_geoms = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(accel_geom_data)];

        let mut accel_build_geom_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .geometries(&accel_geoms)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            accel_struct_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &accel_build_geom_info,
                &[1],
                &mut size_info,
            );
        }

        let accel_struct_buffer_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            size_info.acceleration_structure_size,
            debug_utils,
            None,
            "BLAS",
        );

        let scratch_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            aligned_size!(size_info.build_scratch_size, scratch_buffer_alignment),
            debug_utils,
            Some(scratch_buffer_alignment),
            "BLAS scratch",
        );

        let accel_struct;
        unsafe {
            accel_struct = accel_struct_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(*accel_struct_buffer_resource.vk_buffer())
                        .size(size_info.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
                .unwrap();
        }

        let build_range_info =
            &[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)];

        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] =
            &[build_range_info];

        accel_build_geom_info.dst_acceleration_structure = accel_struct;
        accel_build_geom_info.scratch_data = scratch_buffer.device_or_host_address;

        compute_helper.record_batch(device);
        compute_helper.build_acceleration_structure(
            accel_struct_device,
            &[accel_build_geom_info],
            &build_range_infos,
        );
        compute_helper.submit_batch(device, None, None, None);

        allocator.destroy_buffer_resource(device, scratch_buffer);

        let accel_struct_buffer = *accel_struct_buffer_resource.vk_buffer();
        let accel_struct_allocation = accel_struct_buffer_resource.allocation_type;

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(accel_struct, &(name.to_owned() + " BLAS"));
        }

        Self {
            accel_struct,
            accel_struct_allocation,
            accel_struct_buffer,
        }
    }

    pub fn destroy(
        &mut self,
        device: &ash::Device,
        accel_struct_device: &ash::khr::acceleration_structure::Device,
        allocator: &mut AllocatorType,
    ) {
        allocator.destroy_buffer_and_allocation(
            device,
            self.accel_struct_buffer,
            &mut self.accel_struct_allocation,
        );

        unsafe {
            accel_struct_device.destroy_acceleration_structure(self.accel_struct, None);
        }
    }
}

pub struct TLAS {
    pub accel_struct: vk::AccelerationStructureKHR,
    pub accel_struct_buffer: vk::Buffer,
    pub accel_struct_allocation: vulkan_objects::AllocationType,
}

impl TLAS {
    pub fn new(
        device: &ash::Device,
        allocator: &mut AllocatorType,
        accel_struct_device: &ash::khr::acceleration_structure::Device,
        instances: &[vk::AccelerationStructureInstanceKHR],
        scratch_buffer_alignment: vk::DeviceSize,
        compute_helper: &mut DataHelper,
        debug_utils: &DebugUtils,
        name: &str,
    ) -> Self {
        let staging_instances_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            instances,
            debug_utils,
            None,
            "staging instances",
        );

        let instances_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            (instances.len() * size_of::<vk::AccelerationStructureInstanceKHR>()) as vk::DeviceSize,
            debug_utils,
            None,
            "instances",
        );

        let mut geom_data = vk::AccelerationStructureGeometryDataKHR::default();
        unsafe {
            geom_data.instances = geom_data
                .instances
                .data(instances_buffer.device_or_host_address_const);
        }

        geom_data.instances.s_type =
            vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;

        let geoms = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(geom_data)];

        let mut build_geom_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geoms)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();

        unsafe {
            accel_struct_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geom_info,
                &[instances.len() as u32],
                &mut size_info,
            );
        }

        let accel_struct_buffer_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            size_info.acceleration_structure_size,
            debug_utils,
            None,
            "TLAS",
        );

        let scratch_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            size_info.build_scratch_size,
            debug_utils,
            Some(scratch_buffer_alignment),
            "TLAS scratch",
        );

        let accel_struct;
        unsafe {
            accel_struct = accel_struct_device
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(*accel_struct_buffer_resource.vk_buffer())
                        .size(size_info.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )
                .unwrap();
        }

        let build_range_info = &[vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32)];

        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] =
            &[build_range_info];

        build_geom_info.dst_acceleration_structure = accel_struct;
        build_geom_info.scratch_data = scratch_buffer.device_or_host_address;

        compute_helper.record_batch(device);
        compute_helper.copy_buffer_to_buffer(
            device,
            staging_instances_buffer.vk_buffer(),
            instances_buffer.vk_buffer(),
            staging_instances_buffer.data_size,
        );
        compute_helper.insert_memory_barrier(
            device,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::SHADER_READ,
        );
        compute_helper.build_acceleration_structure(
            accel_struct_device,
            &[build_geom_info],
            &build_range_infos,
        );
        compute_helper.submit_batch(device, None, None, None);

        allocator.destroy_buffer_resource(device, scratch_buffer);
        allocator.destroy_buffer_resource(device, staging_instances_buffer);
        allocator.destroy_buffer_resource(device, instances_buffer);

        let accel_struct_buffer = *accel_struct_buffer_resource.vk_buffer();
        let accel_struct_allocation = accel_struct_buffer_resource.allocation_type;

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(accel_struct, &(name.to_owned() + " TLAS"));
        }

        Self {
            accel_struct,
            accel_struct_allocation,
            accel_struct_buffer,
        }
    }

    pub fn destroy(
        &mut self,
        device: &ash::Device,
        accel_struct_device: &ash::khr::acceleration_structure::Device,
        allocator: &mut AllocatorType,
    ) {
        allocator.destroy_buffer_and_allocation(
            device,
            self.accel_struct_buffer,
            &mut self.accel_struct_allocation,
        );

        unsafe {
            accel_struct_device.destroy_acceleration_structure(self.accel_struct, None);
        }
    }
}
