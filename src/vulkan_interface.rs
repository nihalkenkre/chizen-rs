use crate::vulkan_objects::{self, AllocatorTrait};
use ash::vk;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Place to find the vulkan objects used by other vulkan objects.
/// Allocator can be either GpuAllocatorAllocator or VkMemAllocator
pub struct VulkanInterface<'instance_lifetime> {
    pub instance: vulkan_objects::Instance,
    pub surface_data: vulkan_objects::SurfaceData<'instance_lifetime>,
    pub physical_device_data: vulkan_objects::PhysicalDeviceData<'instance_lifetime>,
    pub device: vulkan_objects::Device,
    pub debug_utils: vulkan_objects::DebugUtils,
    pub swapchain: vulkan_objects::Swapchain,
    pub allocator: vulkan_objects::AllocatorType,
    pub data_helper: vulkan_objects::DataHelper,
}

impl<'instance_lifetime> VulkanInterface<'instance_lifetime> {
    pub fn new(
        instance_required_extenstions: &[*const i8],
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let vk_api_version = vk::make_api_version(0, 1, 4, 341);
        let instance = vulkan_objects::Instance::new(instance_required_extenstions, vk_api_version);
        let mut surface_data =
            vulkan_objects::SurfaceData::new(&instance.instance, display_handle, window_handle);
        let physical_device_data = vulkan_objects::PhysicalDeviceData::new(
            &instance.instance,
            &surface_data.loader,
            &surface_data.surface,
        );

        surface_data.populate_data(&physical_device_data.physical_device);

        let device = vulkan_objects::Device::new(&instance.instance, &physical_device_data);
        let debug_utils = vulkan_objects::DebugUtils::new(&instance.instance, &device.device);

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(instance.instance.handle(), "instance");
            debug_utils.set_object_name(surface_data.surface, "surface");
            debug_utils.set_object_name(physical_device_data.physical_device, "physical Device");
            debug_utils.set_object_name(device.device.handle(), "device");
            debug_utils.set_object_name(device.graphics_queue, "graphics queue");
            debug_utils.set_object_name(device.compute_queue, "compute queue");
            debug_utils.set_object_name(device.transfer_queue, "transfer queue");
        }

        let swapchain = vulkan_objects::Swapchain::new(
            &instance.instance,
            &device.device,
            &surface_data,
            &physical_device_data.queue_family_indices,
            &debug_utils,
        );

        let gpu_allocator_allocator = vulkan_objects::GpuAllocatorAllocator::new(
            &instance.instance,
            &device.device,
            &physical_device_data.physical_device,
        );

        let vk_mem_allocator = vulkan_objects::VkMemAllocator::new(
            &instance.instance,
            &device.device,
            &physical_device_data.physical_device,
            vk_api_version,
        );

        let allocator = vulkan_objects::AllocatorType::VkMem(vk_mem_allocator);
        // let allocator = vulkan_objects::AllocatorType::GpuAllocator(gpu_allocator_allocator);

        let data_helper = vulkan_objects::DataHelper::new(
            &device.device,
            &device.transfer_queue,
            physical_device_data.transfer_queue_family_index,
            &debug_utils,
        );

        Self {
            instance,
            surface_data,
            physical_device_data,
            device,
            debug_utils,
            swapchain,
            allocator,
            data_helper,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        unsafe {
            self.device
                .device
                .queue_wait_idle(self.device.graphics_queue)
                .unwrap();
        }

        self.swapchain.destroy(&self.device.device);
        self.surface_data
            .populate_data(&self.physical_device_data.physical_device);

        let queue_family_indices = [self.physical_device_data.graphics_queue_family_index];
        self.swapchain = vulkan_objects::Swapchain::new(
            &self.instance.instance,
            &self.device.device,
            &self.surface_data,
            &queue_family_indices,
            &self.debug_utils,
        );
    }

    pub fn destroy(self) {
        self.data_helper.destroy(&self.device.device);
        self.allocator.destroy();
        self.swapchain.destroy(&self.device.device);
        self.device.destroy();
        self.surface_data.destroy();
        self.instance.destroy();
    }
}
