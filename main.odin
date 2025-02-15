package main

import "core:image"
import "core:image/png"
import "core:log"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"

import "vendor:glfw"
import vk "vendor:vulkan"

Context :: struct {
	window:                 glfw.WindowHandle,
	instance:               vk.Instance,
	surface:                vk.SurfaceKHR,
	physical_device:        vk.PhysicalDevice,
	device:                 vk.Device,
	queues:                 [Queue_Family]vk.Queue,
	queue_indices:          [Queue_Family]int,
	swapchain:              struct {
		handle:        vk.SwapchainKHR,
		format:        vk.SurfaceFormatKHR,
		present_mode:  vk.PresentModeKHR,
		extent:        vk.Extent2D,
		capabilities:  vk.SurfaceCapabilitiesKHR,
		formats:       []vk.SurfaceFormatKHR,
		present_modes: []vk.PresentModeKHR,
		images:        []vk.Image,
		image_views:   []vk.ImageView,
		framebuffers:  []vk.Framebuffer,
	},
	render_pass:            vk.RenderPass,
	pipeline_layout:        vk.PipelineLayout,
	desc_set_layout:        vk.DescriptorSetLayout,
	pipeline:               vk.Pipeline,
	command_pool:           vk.CommandPool,
	desc_pool:              vk.DescriptorPool,
	command_buffers:        [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer,
	uniform_buffers:        [MAX_FRAMES_IN_FLIGHT]vk.Buffer,
	uniform_buffers_memory: [MAX_FRAMES_IN_FLIGHT]vk.DeviceMemory,
	uniform_buffers_mapped: [MAX_FRAMES_IN_FLIGHT]rawptr,
	image_avail_sems:       [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	render_finished_sems:   [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	in_flight_fences:       [MAX_FRAMES_IN_FLIGHT]vk.Fence,
	descriptor_sets:        [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
	texture:                vk.Image,
	texture_memory:         vk.DeviceMemory,
	texture_image_view:     vk.ImageView,
	vertex_buffer:          vk.Buffer,
	vertex_buffer_memory:   vk.DeviceMemory,
	index_buffer:           vk.Buffer,
	index_buffer_memory:    vk.DeviceMemory,
	current_frame:          uint,
	framebuffer_resized:    bool,
	texture_sampler:        vk.Sampler,
}

Queue_Family :: enum {
	Graphics,
	Present,
}

Vertex :: struct {
	pos:   [2]f32,
	color: [3]f32,
    tex:  [2]f32
}

Uniform_Buffer_Object :: struct {
	model: matrix[4, 4]f32,
	view:  matrix[4, 4]f32,
	proj:  matrix[4, 4]f32,
}

VALIDATION_LAYERS :: []cstring{"VK_LAYER_KHRONOS_validation"}

DEVICE_EXTENSIONS :: []cstring{"VK_KHR_swapchain"}

MAX_FRAMES_IN_FLIGHT :: 2

INIT_WIDTH :: 640
INIT_HEIGHT :: 480
APP_TITLE :: "Vulkaaan"

vertices := []Vertex {
	{{-0.5, -0.5}, {0.0, 0.0, 1.0}, {0.0, 0.0}},
	{{0.5, -0.5}, {1.0, 1.0, 0.0}, {1.0, 0.0}},
	{{0.5, 0.5}, {0.0, 1.0, 1.0}, {1.0, 1.0}},
	{{-0.5, 0.5}, {1.0, 0.0, 0.0}, {0.0, 1.0}},
}

indices := []u16{0, 1, 2, 0, 2, 3}

window_init :: proc() {
	framebuffer_size_cb :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
		ctx := cast(^Context)glfw.GetWindowUserPointer(window)
		ctx.framebuffer_resized = true
	}

	ctx := cast(^Context)context.user_ptr

	glfw.Init()

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, false)

	ctx.window = glfw.CreateWindow(INIT_WIDTH, INIT_HEIGHT, APP_TITLE, nil, nil)
	glfw.SetWindowUserPointer(ctx.window, ctx)
	glfw.SetFramebufferSizeCallback(ctx.window, framebuffer_size_cb)
}

window_deinit :: proc() {
	ctx := cast(^Context)context.user_ptr

	glfw.DestroyWindow(ctx.window)
	glfw.Terminate()
}

vk_init :: proc() {
	using ctx := cast(^Context)context.user_ptr

	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))
	vk_create_instance()
	vk.load_proc_addresses(instance)

	glfw.CreateWindowSurface(instance, window, nil, &surface)
	vk_create_device()

	for &q, f in queues do vk.GetDeviceQueue(device, u32(queue_indices[f]), 0, &q)

	vk_create_swapchain()
	vk_create_image_views()
	vk_create_renderpass()
	vk_create_desc_set_layout()
	vk_create_pipeline()
	vk_create_framebuffers()
	vk_create_command_pool()
	vk_create_texture_image()
	vk_create_texture_image_view()
	vk_create_texture_sampler()
	vk_create_vertex_buffer()
	vk_create_index_buffer()
	vk_create_uniform_buffers()
	vk_create_descriptor_pool()
	vk_create_descriptor_sets()
	vk_create_sync_objects()
}

vk_deinit :: proc() {
	using ctx := cast(^Context)context.user_ptr

	vk.DestroyBuffer(device, index_buffer, nil)
	vk.FreeMemory(device, index_buffer_memory, nil)
	vk.DestroyBuffer(device, vertex_buffer, nil)
	vk.FreeMemory(device, vertex_buffer_memory, nil)

	vk_cleanup_swapchain()

	vk.DestroySampler(device, texture_sampler, nil)
	vk.DestroyImageView(device, texture_image_view, nil)

	vk.DestroyImage(device, texture, nil)
	vk.FreeMemory(device, texture_memory, nil)

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		vk.DestroyBuffer(device, uniform_buffers[i], nil)
		vk.FreeMemory(device, uniform_buffers_memory[i], nil)
	}

	vk.DestroyDescriptorPool(device, desc_pool, nil)

	vk.DestroyDescriptorSetLayout(device, desc_set_layout, nil)

	vk.DestroyCommandPool(device, command_pool, nil)

	vk.DestroyPipeline(device, pipeline, nil)
	vk.DestroyPipelineLayout(device, pipeline_layout, nil)
	vk.DestroyRenderPass(device, render_pass, nil)

	vk.DestroyDevice(device, nil)
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyInstance(instance, nil)
}

vk_create_instance :: proc() {
	using ctx := cast(^Context)context.user_ptr

	create_info := vk.InstanceCreateInfo {
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &{
			sType = .APPLICATION_INFO,
			pApplicationName = APP_TITLE,
			applicationVersion = vk.MAKE_VERSION(0, 0, 1),
			pEngineName = "No engine",
			engineVersion = vk.MAKE_VERSION(0, 0, 1),
			apiVersion = vk.API_VERSION_1_0,
		},
	}

	exts_required := glfw.GetRequiredInstanceExtensions()
	exts_avail_n: u32
	vk_must(vk.EnumerateInstanceExtensionProperties(nil, &exts_avail_n, nil))
	exts_avail := make([]vk.ExtensionProperties, exts_avail_n, context.temp_allocator)
	vk_must(vk.EnumerateInstanceExtensionProperties(nil, &exts_avail_n, raw_data(exts_avail)))

	exts_outer: for ext_required in exts_required {
		er_name := string(ext_required)
		for &ext_avail in exts_avail {
			ea_name := byte_arr_str(&ext_avail.extensionName)
			if strings.compare(er_name, ea_name) == 0 do continue exts_outer
		}

		log.panicf("vk: extension %q not available\n", er_name)
	}

	create_info.enabledExtensionCount = u32(len(exts_required))
	create_info.ppEnabledExtensionNames = raw_data(exts_required)

	when ODIN_DEBUG {
		layers_avail_n: u32
		vk_must(vk.EnumerateInstanceLayerProperties(&layers_avail_n, nil))
		layers_avail := make([]vk.LayerProperties, layers_avail_n, context.temp_allocator)
		vk_must(vk.EnumerateInstanceLayerProperties(&layers_avail_n, raw_data(layers_avail)))

		layers_outer: for layer in VALIDATION_LAYERS {
			layer_name := string(layer)
			for &layer_avail in layers_avail {
				la_name := byte_arr_str(&layer_avail.layerName)
				if strings.compare(layer_name, la_name) == 0 do continue layers_outer
			}

			log.panicf("vk: validation layer %q not available\n", layer_name)
		}

		create_info.enabledLayerCount = u32(len(VALIDATION_LAYERS))
		create_info.ppEnabledLayerNames = raw_data(VALIDATION_LAYERS)
	} else {
		create_info.enabledLayerCount = 0
	}

	vk_must(vk.CreateInstance(&create_info, nil, &instance))
}

vk_create_device :: proc() {
	using ctx := (^Context)(context.user_ptr)

	phys_devices_n: u32
	vk_must(vk.EnumeratePhysicalDevices(instance, &phys_devices_n, nil))
	phys_devices := make([]vk.PhysicalDevice, phys_devices_n, context.temp_allocator)
	vk_must(vk.EnumeratePhysicalDevices(instance, &phys_devices_n, raw_data(phys_devices)))

	hiscore := -1

	devs_outer: for phys_device in phys_devices {
		exts_avail_n: u32
		if vk.EnumerateDeviceExtensionProperties(phys_device, nil, &exts_avail_n, nil) != .SUCCESS do continue

		exts_avail := make([]vk.ExtensionProperties, exts_avail_n, context.temp_allocator)
		if vk.EnumerateDeviceExtensionProperties(phys_device, nil, &exts_avail_n, raw_data(exts_avail)) != .SUCCESS do continue

		exts_outer: for ext in DEVICE_EXTENSIONS {
			e_name := string(ext)
			for &ext_avail in exts_avail {
				ea_name := byte_arr_str(&ext_avail.extensionName)
				if strings.compare(e_name, ea_name) == 0 do continue exts_outer
			}

			continue devs_outer
		}

		props: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(phys_device, &props)

		feats: vk.PhysicalDeviceFeatures
		vk.GetPhysicalDeviceFeatures(phys_device, &feats)

		if !feats.samplerAnisotropy do continue

		vk_query_swapchain_details(phys_device)
		if len(swapchain.formats) == 0 || len(swapchain.present_modes) == 0 do continue

		vk_query_queue_families(phys_device)
		for q in queue_indices do if q == -1 do continue

		score := 0

		if props.deviceType == .DISCRETE_GPU {
			score += 3000
		} else if props.deviceType == .INTEGRATED_GPU {
			score += 2000
		} else if props.deviceType == .VIRTUAL_GPU {
			score += 1000
		}

		score += int(props.limits.maxImageDimension2D)

		if score > hiscore {
			hiscore = score
			physical_device = phys_device
		}
	}

	if hiscore == -1 {
		log.panic("vk: could not choose a gpu\n")
	}

	unique_indices := make(map[int]b8, context.temp_allocator)
	for i in queue_indices do unique_indices[i] = true

	queue_create_infos := make(
		[]vk.DeviceQueueCreateInfo,
		len(unique_indices),
		context.temp_allocator,
	)
	queue_priority: f32 = 1.0

	for k in unique_indices {
		queue_create_infos[k] = vk.DeviceQueueCreateInfo {
			sType            = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = u32(k),
			queueCount       = 1,
			pQueuePriorities = &queue_priority,
		}
	}

	features := vk.PhysicalDeviceFeatures {
		samplerAnisotropy = true,
	}

	create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		enabledExtensionCount   = u32(len(DEVICE_EXTENSIONS)),
		ppEnabledExtensionNames = raw_data(DEVICE_EXTENSIONS),
		queueCreateInfoCount    = u32(len(queue_create_infos)),
		pQueueCreateInfos       = raw_data(queue_create_infos),
		pEnabledFeatures        = &features,
	}

	vk_must(vk.CreateDevice(physical_device, &create_info, nil, &device))
}

vk_query_swapchain_details :: proc(phys_device: vk.PhysicalDevice) {
	using ctx := (^Context)(context.user_ptr)

	vk_must(
		vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device, surface, &swapchain.capabilities),
	)

	fmts_n: u32
	vk_must(vk.GetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &fmts_n, nil))
	if (fmts_n > 0) {
		swapchain.formats = make([]vk.SurfaceFormatKHR, fmts_n)
		vk_must(
			vk.GetPhysicalDeviceSurfaceFormatsKHR(
				phys_device,
				surface,
				&fmts_n,
				raw_data(swapchain.formats),
			),
		)
	}

	modes_n: u32
	vk_must(vk.GetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &modes_n, nil))
	if (modes_n > 0) {
		swapchain.present_modes = make([]vk.PresentModeKHR, modes_n)
		vk_must(
			vk.GetPhysicalDeviceSurfacePresentModesKHR(
				phys_device,
				surface,
				&modes_n,
				raw_data(swapchain.present_modes),
			),
		)
	}
}

vk_query_queue_families :: proc(phys_device: vk.PhysicalDevice) {
	using ctx := (^Context)(context.user_ptr)

	fams_n: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(phys_device, &fams_n, nil)
	fams := make([]vk.QueueFamilyProperties, fams_n, context.temp_allocator)
	vk.GetPhysicalDeviceQueueFamilyProperties(phys_device, &fams_n, raw_data(fams))

	for &q in queue_indices do q = -1

	for fam, i in fams {
		if .GRAPHICS in fam.queueFlags && queue_indices[.Graphics] == -1 do queue_indices[.Graphics] = i

		supported: b32
		vk_must(vk.GetPhysicalDeviceSurfaceSupportKHR(phys_device, u32(i), surface, &supported))
		if supported && queue_indices[.Present] == -1 do queue_indices[.Present] = i

		for q in queue_indices do if q == -1 do continue
		break
	}
}

vk_create_swapchain :: proc() {
	using ctx := (^Context)(context.user_ptr)

	swapchain.format = swapchain.formats[0]
	for f in swapchain.formats {
		if f.format == .B8G8R8A8_SRGB && f.colorSpace == .SRGB_NONLINEAR do swapchain.format = f
	}

	swapchain.present_mode = .FIFO
	for m in swapchain.present_modes {
		if m == .MAILBOX do swapchain.present_mode = m
	}

	if (swapchain.capabilities.currentExtent.width != max(u32)) {
		swapchain.extent = swapchain.capabilities.currentExtent
	} else {
		w, h := glfw.GetFramebufferSize(window)
		swapchain.extent.width = clamp(
			u32(w),
			swapchain.capabilities.minImageExtent.width,
			swapchain.capabilities.maxImageExtent.height,
		)
		swapchain.extent.height = clamp(
			u32(h),
			swapchain.capabilities.minImageExtent.height,
			swapchain.capabilities.maxImageExtent.height,
		)
	}

	images_n := swapchain.capabilities.minImageCount + 1
	if (swapchain.capabilities.maxImageCount > 0 &&
		   images_n > swapchain.capabilities.maxImageCount) {
		images_n = swapchain.capabilities.maxImageCount
	}

	create_info := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = surface,
		minImageCount    = images_n,
		imageFormat      = swapchain.format.format,
		imageColorSpace  = swapchain.format.colorSpace,
		imageExtent      = swapchain.extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
		preTransform     = swapchain.capabilities.currentTransform,
		compositeAlpha   = {.OPAQUE},
		presentMode      = swapchain.present_mode,
		clipped          = true,
	}

	if (queue_indices[.Graphics] != queue_indices[.Present]) {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = raw_data(
			[]u32{u32(queue_indices[.Graphics]), u32(queue_indices[.Present])},
		)
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
	}

	vk_must(vk.CreateSwapchainKHR(device, &create_info, nil, &swapchain.handle))
}

vk_recreate_swapchain :: proc() {
	using ctx := (^Context)(context.user_ptr)

	width, height := glfw.GetFramebufferSize(window)
	for width == 0 && height == 0 {
		width, height = glfw.GetFramebufferSize(window)
		glfw.WaitEvents()
	}

	vk.DeviceWaitIdle(device)

	vk_cleanup_swapchain()

	vk_create_swapchain()
	vk_create_image_views()
	vk_create_framebuffers()
	vk_create_sync_objects()
}

vk_cleanup_swapchain :: proc() {
	using ctx := (^Context)(context.user_ptr)

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		vk.DestroySemaphore(device, render_finished_sems[i], nil)
		vk.DestroySemaphore(device, image_avail_sems[i], nil)
		vk.DestroyFence(device, in_flight_fences[i], nil)
	}

	for f in swapchain.framebuffers do vk.DestroyFramebuffer(device, f, nil)

	for view in swapchain.image_views do vk.DestroyImageView(device, view, nil)

	vk.DestroySwapchainKHR(device, swapchain.handle, nil)
}

vk_create_image_views :: proc() {
	using ctx := (^Context)(context.user_ptr)

	images_n: u32
	vk_must(vk.GetSwapchainImagesKHR(device, swapchain.handle, &images_n, nil))

	swapchain.images = make([]vk.Image, images_n)
	vk_must(
		vk.GetSwapchainImagesKHR(device, swapchain.handle, &images_n, raw_data(swapchain.images)),
	)

	swapchain.image_views = make([]vk.ImageView, images_n)

	for image, i in swapchain.images {
		swapchain.image_views[i] = vk_create_image_view(image, swapchain.format.format)
	}
}

vk_create_renderpass :: proc() {
	using ctx := (^Context)(context.user_ptr)

	color_attachment := vk.AttachmentDescription {
		format         = swapchain.format.format,
		samples        = {._1},
		loadOp         = .CLEAR,
		storeOp        = .STORE,
		stencilLoadOp  = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout  = .UNDEFINED,
		finalLayout    = .PRESENT_SRC_KHR,
	}

	color_attachment_ref := vk.AttachmentReference {
		attachment = 0,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint    = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment_ref,
	}

	subpass_dep := vk.SubpassDependency {
		srcSubpass    = vk.SUBPASS_EXTERNAL,
		dstSubpass    = 0,
		srcStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {},
		dstStageMask  = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
	}

	create_info := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_attachment,
		subpassCount    = 1,
		pSubpasses      = &subpass,
		dependencyCount = 1,
		pDependencies   = &subpass_dep,
	}

	vk_must(vk.CreateRenderPass(device, &create_info, nil, &render_pass))
}

vk_create_desc_set_layout :: proc() {
	using ctx := (^Context)(context.user_ptr)

	ubo_layout := vk.DescriptorSetLayoutBinding {
		binding         = 0,
		descriptorType  = .UNIFORM_BUFFER,
		descriptorCount = 1,
		stageFlags      = {.VERTEX},
	}

	sampler_layout := vk.DescriptorSetLayoutBinding{
        binding = 1,
        descriptorCount = 1,
        descriptorType = .COMBINED_IMAGE_SAMPLER,
        pImmutableSamplers = nil,
        stageFlags = {.FRAGMENT}
    }

    bindings := [?]vk.DescriptorSetLayoutBinding{ubo_layout, sampler_layout}

	create_info := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		bindingCount = 2,
		pBindings    = raw_data(&bindings),
	}

	vk_must(vk.CreateDescriptorSetLayout(device, &create_info, nil, &desc_set_layout))
}

vk_create_pipeline :: proc() {
	using ctx := (^Context)(context.user_ptr)

	vs_code, vs_ok := os.read_entire_file("shaders/shader.vert.spv")
	if !vs_ok do log.panic("io: could not create vertex shader")

	fs_code, fs_ok := os.read_entire_file("shaders/shader.frag.spv")
	if !fs_ok do log.panic("io: could not create fragment shader")

	vs_shader := vk_create_shader_module(vs_code)
	defer vk.DestroyShaderModule(device, vs_shader, nil)

	fs_shader := vk_create_shader_module(fs_code)
	defer vk.DestroyShaderModule(device, fs_shader, nil)

	vs_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.VERTEX},
		module = vs_shader,
		pName  = "main",
	}

	fs_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.FRAGMENT},
		module = fs_shader,
		pName  = "main",
	}

	shader_stages := []vk.PipelineShaderStageCreateInfo{vs_info, fs_info}

	binding_desc := vk_vertex_binding_desc()
	attr_descs := vk_vertex_attr_descs()

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		vertexAttributeDescriptionCount = len(attr_descs),
		pVertexBindingDescriptions      = &binding_desc,
		pVertexAttributeDescriptions    = raw_data(attr_descs[:]),
	}

	input_assembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport := vk.Viewport {
		x        = 0.0,
		y        = 0.0,
		width    = f32(swapchain.extent.width),
		height   = f32(swapchain.extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = swapchain.extent,
	}

	dynamic_states := []vk.DynamicState{.VIEWPORT, .SCISSOR}
	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates    = raw_data(dynamic_states),
	}

	viewport_state := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType                   = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable        = false,
		rasterizerDiscardEnable = false,
		polygonMode             = .FILL,
		lineWidth               = 1.0,
		cullMode                = {.BACK},
		frontFace               = .COUNTER_CLOCKWISE,
		depthBiasEnable         = false,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		sampleShadingEnable  = false,
		rasterizationSamples = {._1},
	}

	colorblend_attachment := vk.PipelineColorBlendAttachmentState {
		blendEnable         = true,
		srcColorBlendFactor = .SRC_ALPHA,
		dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
		colorBlendOp        = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ZERO,
		alphaBlendOp        = .ADD,
		colorWriteMask      = {.R, .G, .B, .A},
	}

	color_blending := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable   = false,
		attachmentCount = 1,
		pAttachments    = &colorblend_attachment,
	}

	pipeline_layout_create_info := vk.PipelineLayoutCreateInfo {
		sType          = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = 1,
		pSetLayouts    = &desc_set_layout,
	}

	vk_must(vk.CreatePipelineLayout(device, &pipeline_layout_create_info, nil, &pipeline_layout))

	create_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = 2,
		pStages             = raw_data(shader_stages),
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &input_assembly,
		pViewportState      = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState   = &multisampling,
		pColorBlendState    = &color_blending,
		pDynamicState       = &dynamic_state,
		layout              = pipeline_layout,
		renderPass          = render_pass,
		subpass             = 0,
	}

	vk_must(vk.CreateGraphicsPipelines(device, 0, 1, &create_info, nil, &pipeline))
}

vk_create_shader_module :: proc(code: []byte) -> vk.ShaderModule {
	using ctx := (^Context)(context.user_ptr)

	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		pCode    = (^u32)(raw_data(code)),
		codeSize = len(code),
	}

	shader_module: vk.ShaderModule

	vk_must(vk.CreateShaderModule(device, &create_info, nil, &shader_module))

	return shader_module
}

vk_create_framebuffers :: proc() {
	using ctx := (^Context)(context.user_ptr)

	swapchain.framebuffers = make([]vk.Framebuffer, len(swapchain.image_views))
	for &view, i in swapchain.image_views {
		create_info := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = render_pass,
			attachmentCount = 1,
			pAttachments    = &view,
			width           = swapchain.extent.width,
			height          = swapchain.extent.height,
			layers          = 1,
		}

		vk_must(vk.CreateFramebuffer(device, &create_info, nil, &swapchain.framebuffers[i]))
	}
}

vk_create_texture_image :: proc() {
	using ctx := (^Context)(context.user_ptr)

	img, err := image.load("textures/texture.png", {.alpha_add_if_missing})
	if err != nil {
		log.panic("io: could not load texture")
	}
	defer delete(img.pixels.buf)

	assert(img.channels == 4 && img.depth == 8)

	size := cast(vk.DeviceSize)(img.width * img.height * 4)

	staging_buf: vk.Buffer
	staging_mem: vk.DeviceMemory

	vk_create_buffer(
		size,
		{.TRANSFER_SRC},
		{.HOST_COHERENT, .HOST_VISIBLE},
		&staging_buf,
		&staging_mem,
	)

	data: rawptr
	vk.MapMemory(device, staging_mem, 0, size, {}, &data)
	mem.copy(data, raw_data(img.pixels.buf), int(size))
	vk.UnmapMemory(device, staging_mem)

	vk_create_image(
		u32(img.width),
		u32(img.height),
		.R8G8B8A8_SRGB,
		.OPTIMAL,
		{.TRANSFER_DST, .SAMPLED},
		{.DEVICE_LOCAL},
		&texture,
		&texture_memory,
	)

	vk_transition_image_layout(texture, .R8G8B8A8_SRGB, .UNDEFINED, .TRANSFER_DST_OPTIMAL)
	vk_copy_buffer_to_image(staging_buf, texture, u32(img.width), u32(img.height))
	vk_transition_image_layout(
		texture,
		.R8G8B8A8_SRGB,
		.TRANSFER_DST_OPTIMAL,
		.SHADER_READ_ONLY_OPTIMAL,
	)

	vk.DestroyBuffer(device, staging_buf, nil)
	vk.FreeMemory(device, staging_mem, nil)
}

vk_create_texture_image_view :: proc() {
	using ctx := (^Context)(context.user_ptr)
	texture_image_view = vk_create_image_view(texture, .R8G8B8A8_SRGB)
}

vk_create_texture_sampler :: proc() {
	using ctx := (^Context)(context.user_ptr)

	dev_props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physical_device, &dev_props)

	sampler_info := vk.SamplerCreateInfo {
		sType                   = .SAMPLER_CREATE_INFO,
		magFilter               = .LINEAR,
		minFilter               = .LINEAR,
		addressModeU            = .REPEAT,
		addressModeV            = .REPEAT,
		addressModeW            = .REPEAT,
		anisotropyEnable        = true,
		maxAnisotropy           = dev_props.limits.maxSamplerAnisotropy,
		borderColor             = .INT_OPAQUE_BLACK,
		unnormalizedCoordinates = false,
		compareEnable           = false,
		compareOp               = .ALWAYS,
		mipmapMode              = .LINEAR,
		mipLodBias              = 0.0,
		minLod                  = 0.0,
		maxLod                  = 0.0,
	}

	vk_must(vk.CreateSampler(device, &sampler_info, nil, &texture_sampler))
}

vk_create_image_view :: proc(image: vk.Image, format: vk.Format) -> (image_view: vk.ImageView) {
	using ctx := (^Context)(context.user_ptr)

	view_info := vk.ImageViewCreateInfo {
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = image,
		viewType = .D2,
		format = format,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	vk_must(vk.CreateImageView(device, &view_info, nil, &image_view))
	return
}

vk_create_image :: proc(
	width, height: u32,
	format: vk.Format,
	tiling: vk.ImageTiling,
	usage: vk.ImageUsageFlags,
	props: vk.MemoryPropertyFlags,
	image: ^vk.Image,
	image_mem: ^vk.DeviceMemory,
) {
	using ctx := (^Context)(context.user_ptr)

	image_info := vk.ImageCreateInfo {
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		extent = {width = width, height = height, depth = 1},
		mipLevels = 1,
		arrayLayers = 1,
		format = format,
		tiling = tiling,
		initialLayout = .UNDEFINED,
		usage = usage,
		sharingMode = .EXCLUSIVE,
		samples = {._1},
	}

	vk_must(vk.CreateImage(device, &image_info, nil, image))

	mem_reqs: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, image^, &mem_reqs)

	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_reqs.size,
		memoryTypeIndex = vk_find_memory_type(mem_reqs.memoryTypeBits, props),
	}

	vk_must(vk.AllocateMemory(device, &alloc_info, nil, image_mem))
	vk.BindImageMemory(device, image^, image_mem^, 0)
}

vk_transition_image_layout :: proc(
	image: vk.Image,
	format: vk.Format,
	old_layout: vk.ImageLayout,
	new_layout: vk.ImageLayout,
) {
	cmd_buf := vk_begin_one_time_cmds()

	barrier := vk.ImageMemoryBarrier {
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = old_layout,
		newLayout = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = {},
		dstAccessMask = {},
	}

	src_stage: vk.PipelineStageFlags
	dst_stage: vk.PipelineStageFlags

	if old_layout == .UNDEFINED && new_layout == .TRANSFER_DST_OPTIMAL {
		barrier.srcAccessMask = {}
		barrier.dstAccessMask = {.TRANSFER_WRITE}

		src_stage = {.TOP_OF_PIPE}
		dst_stage = {.TRANSFER}
	} else if old_layout == .TRANSFER_DST_OPTIMAL && new_layout == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}

		src_stage = {.TRANSFER}
		dst_stage = {.FRAGMENT_SHADER}
	} else {
		log.panic("unsupported layout transition")
	}

	vk.CmdPipelineBarrier(cmd_buf, src_stage, dst_stage, {}, 0, nil, 0, nil, 1, &barrier)

	vk_end_one_time_cmds(&cmd_buf)
}

vk_create_vertex_buffer :: proc() {
	using ctx := (^Context)(context.user_ptr)

	size := cast(vk.DeviceSize)slice.size(vertices)

	staging: vk.Buffer
	staging_mem: vk.DeviceMemory

	vk_create_buffer(
		size,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&staging,
		&staging_mem,
	)

	data: rawptr
	vk_must(vk.MapMemory(device, staging_mem, 0, size, {}, &data))
	mem.copy(data, raw_data(vertices), int(size))
	vk.UnmapMemory(device, staging_mem)

	vk_create_buffer(
		size,
		{.VERTEX_BUFFER, .TRANSFER_DST},
		{.DEVICE_LOCAL},
		&vertex_buffer,
		&vertex_buffer_memory,
	)

	vk_copy_buffer(staging, vertex_buffer, size)

	vk.DestroyBuffer(device, staging, nil)
	vk.FreeMemory(device, staging_mem, nil)
}

vk_create_index_buffer :: proc() {
	using ctx := (^Context)(context.user_ptr)

	size := cast(vk.DeviceSize)slice.size(indices)

	staging: vk.Buffer
	staging_mem: vk.DeviceMemory

	vk_create_buffer(
		size,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
		&staging,
		&staging_mem,
	)

	data: rawptr
	vk_must(vk.MapMemory(device, staging_mem, 0, size, {}, &data))
	mem.copy(data, raw_data(indices), int(size))
	vk.UnmapMemory(device, staging_mem)

	vk_create_buffer(
		size,
		{.INDEX_BUFFER, .TRANSFER_DST},
		{.DEVICE_LOCAL},
		&index_buffer,
		&index_buffer_memory,
	)

	vk_copy_buffer(staging, index_buffer, size)

	vk.DestroyBuffer(device, staging, nil)
	vk.FreeMemory(device, staging_mem, nil)
}

vk_create_uniform_buffers :: proc() {
	using ctx := (^Context)(context.user_ptr)

	size := cast(vk.DeviceSize)size_of(Uniform_Buffer_Object)

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		vk_create_buffer(
			size,
			{.UNIFORM_BUFFER},
			{.HOST_VISIBLE, .HOST_COHERENT},
			&uniform_buffers[i],
			&uniform_buffers_memory[i],
		)
		vk.MapMemory(device, uniform_buffers_memory[i], 0, size, {}, &uniform_buffers_mapped[i])
	}
}

vk_create_descriptor_pool :: proc() {
	using ctx := (^Context)(context.user_ptr)

    pool_sizes := [?]vk.DescriptorPoolSize{
        {
            type            = .UNIFORM_BUFFER,
            descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
        {
            type            = .COMBINED_IMAGE_SAMPLER,
            descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
    }

	create_info := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		poolSizeCount = len(pool_sizes),
		pPoolSizes    = raw_data(&pool_sizes),
		maxSets       = MAX_FRAMES_IN_FLIGHT,
	}

	vk_must(vk.CreateDescriptorPool(device, &create_info, nil, &desc_pool))
}

vk_create_descriptor_sets :: proc() {
	using ctx := (^Context)(context.user_ptr)

	layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{desc_set_layout, desc_set_layout}

	alloc_info := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = desc_pool,
		descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
		pSetLayouts        = &layouts[0],
	}

	vk_must(vk.AllocateDescriptorSets(device, &alloc_info, &descriptor_sets[0]))

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		buffer_info := vk.DescriptorBufferInfo {
			buffer = uniform_buffers[i],
			offset = 0,
			range  = size_of(Uniform_Buffer_Object),
		}

        image_info := vk.DescriptorImageInfo {
            imageLayout = .SHADER_READ_ONLY_OPTIMAL,
            imageView = texture_image_view,
            sampler = texture_sampler
        }

        desc_writes := [?]vk.WriteDescriptorSet{
            {
                sType           = .WRITE_DESCRIPTOR_SET,
                dstSet          = descriptor_sets[i],
                dstBinding      = 0,
                dstArrayElement = 0,
                descriptorType  = .UNIFORM_BUFFER,
                descriptorCount = 1,
                pBufferInfo     = &buffer_info,
            },
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = descriptor_sets[i],
                dstBinding = 1,
                dstArrayElement = 0,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = 1,
                pImageInfo = &image_info
            }
        }

		vk.UpdateDescriptorSets(device, len(desc_writes), raw_data(&desc_writes), 0, nil)
	}
}

vk_create_command_pool :: proc() {
	using ctx := (^Context)(context.user_ptr)

	pool_info := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = u32(queue_indices[.Graphics]),
	}

	vk_must(vk.CreateCommandPool(device, &pool_info, nil, &command_pool))

	alloc_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = command_pool,
		level              = .PRIMARY,
		commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	}

	vk_must(vk.AllocateCommandBuffers(device, &alloc_info, &command_buffers[0]))
}

vk_create_sync_objects :: proc() {
	using ctx := (^Context)(context.user_ptr)

	sem_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}

	fence_info := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}

	for i := 0; i < MAX_FRAMES_IN_FLIGHT; i += 1 {
		vk_must(vk.CreateSemaphore(device, &sem_info, nil, &image_avail_sems[i]))
		vk_must(vk.CreateSemaphore(device, &sem_info, nil, &render_finished_sems[i]))
		vk_must(vk.CreateFence(device, &fence_info, nil, &in_flight_fences[i]))
	}
}

vk_record_command_buffer :: proc(cmd_buf: vk.CommandBuffer, img_idx: u32) {
	using ctx := (^Context)(context.user_ptr)

	cmd_buf_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
	}

	vk_must(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_info))

	render_pass_info := vk.RenderPassBeginInfo {
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = render_pass,
		framebuffer = swapchain.framebuffers[img_idx],
		renderArea = {offset = {0, 0}, extent = swapchain.extent},
		clearValueCount = 1,
		pClearValues = &vk.ClearValue{color = {int32 = {0, 0, 0, 0}}},
	}

	vk.CmdBeginRenderPass(cmd_buf, &render_pass_info, .INLINE)

	vk.CmdBindPipeline(cmd_buf, .GRAPHICS, pipeline)

	vk.CmdBindVertexBuffers(cmd_buf, 0, 1, &vertex_buffer, raw_data([]vk.DeviceSize{0}))
	vk.CmdBindIndexBuffer(cmd_buf, index_buffer, 0, .UINT16)

	viewport := vk.Viewport {
		x        = 0.0,
		y        = 0.0,
		width    = f32(swapchain.extent.width),
		height   = f32(swapchain.extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}

	vk.CmdSetViewport(cmd_buf, 0, 1, &viewport)

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = swapchain.extent,
	}

	vk.CmdSetScissor(cmd_buf, 0, 1, &scissor)

	vk.CmdBindDescriptorSets(
		cmd_buf,
		.GRAPHICS,
		pipeline_layout,
		0,
		1,
		&descriptor_sets[current_frame],
		0,
		nil,
	)
	vk.CmdDrawIndexed(cmd_buf, u32(len(indices)), 1, 0, 0, 0)

	vk.CmdEndRenderPass(cmd_buf)

	vk_must(vk.EndCommandBuffer(cmd_buf))
}

vk_vertex_binding_desc :: proc() -> vk.VertexInputBindingDescription {
	return vk.VertexInputBindingDescription {
		binding = 0,
		stride = size_of(Vertex),
		inputRate = .VERTEX,
	}
}

vk_vertex_attr_descs :: proc() -> [3]vk.VertexInputAttributeDescription {
	return [3]vk.VertexInputAttributeDescription {
		vk.VertexInputAttributeDescription {
			binding = 0,
			location = 0,
			format = .R32G32_SFLOAT,
			offset = u32(offset_of(Vertex, pos)),
		},
		vk.VertexInputAttributeDescription {
			binding = 0,
			location = 1,
			format = .R32G32B32_SFLOAT,
			offset = u32(offset_of(Vertex, color)),
		},
		vk.VertexInputAttributeDescription {
			binding = 0,
			location = 2,
			format = .R32G32_SFLOAT,
			offset = u32(offset_of(Vertex, tex)),
		},
	}
}

vk_find_memory_type :: proc(type_filter: u32, props: vk.MemoryPropertyFlags) -> u32 {
	using ctx := (^Context)(context.user_ptr)

	mem_props: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &mem_props)

	for i: u32 = 0; i < mem_props.memoryTypeCount; i += 1 {
		if (type_filter & (1 << i) != 0 &&
			   mem_props.memoryTypes[i].propertyFlags & props == props) {
			return i
		}
	}

	log.panic("vk: failed to find memory type")
}

vk_create_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	props: vk.MemoryPropertyFlags,
	buffer: ^vk.Buffer,
	memory: ^vk.DeviceMemory,
) {
	using ctx := (^Context)(context.user_ptr)

	create_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}

	vk_must(vk.CreateBuffer(device, &create_info, nil, buffer))

	mem_reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, buffer^, &mem_reqs)

	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = mem_reqs.size,
		memoryTypeIndex = vk_find_memory_type(mem_reqs.memoryTypeBits, props),
	}

	vk_must(vk.AllocateMemory(device, &alloc_info, nil, memory))
	vk_must(vk.BindBufferMemory(device, buffer^, memory^, 0))
}

vk_begin_one_time_cmds :: proc() -> (cmd_buf: vk.CommandBuffer) {
	using ctx := (^Context)(context.user_ptr)

	alloc_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		level              = .PRIMARY,
		commandPool        = command_pool,
		commandBufferCount = 1,
	}

	vk_must(vk.AllocateCommandBuffers(device, &alloc_info, &cmd_buf))

	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}

	vk_must(vk.BeginCommandBuffer(cmd_buf, &begin_info))

	return
}

vk_end_one_time_cmds :: proc(cmd_buf: ^vk.CommandBuffer) {
	using ctx := (^Context)(context.user_ptr)

	vk_must(vk.EndCommandBuffer(cmd_buf^))

	submit_info := vk.SubmitInfo {
		sType              = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers    = cmd_buf,
	}

	vk.QueueSubmit(queues[.Graphics], 1, &submit_info, {})
	vk.QueueWaitIdle(queues[.Graphics])

	vk.FreeCommandBuffers(device, command_pool, 1, cmd_buf)
}

vk_copy_buffer :: proc(src: vk.Buffer, dst: vk.Buffer, size: vk.DeviceSize) {
	using ctx := (^Context)(context.user_ptr)

	cmd_buf := vk_begin_one_time_cmds()

	copy_region := vk.BufferCopy {
		size = size,
	}

	vk.CmdCopyBuffer(cmd_buf, src, dst, 1, &copy_region)

	vk_end_one_time_cmds(&cmd_buf)
}

vk_copy_buffer_to_image :: proc(buffer: vk.Buffer, image: vk.Image, width: u32, height: u32) {
	using ctx := (^Context)(context.user_ptr)

	cmd_buf := vk_begin_one_time_cmds()

	copy_region := vk.BufferImageCopy {
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,
		imageSubresource = {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		imageOffset = {0, 0, 0},
		imageExtent = {width, height, 1},
	}

	vk.CmdCopyBufferToImage(cmd_buf, buffer, image, .TRANSFER_DST_OPTIMAL, 1, &copy_region)

	vk_end_one_time_cmds(&cmd_buf)
}

vk_update_uniform_buffer :: proc() {
	@(static) first := true
	@(static) start: time.Tick

	using ctx := (^Context)(context.user_ptr)

	if (first) {
		start = time.tick_now()
		first = false
	}

	dur := f32(time.tick_since(start))

	ubo := Uniform_Buffer_Object {
		model = linalg.matrix4_rotate(
			dur * linalg.to_radians(f32(90.0)) * 0.000000001,
			[3]f32{0.0, 0.0, 1.0},
		),
		view  = linalg.matrix4_look_at([3]f32{2.0, 2.0, 2.0}, [3]f32{}, [3]f32{0.0, 0.0, 1.0}),
		proj  = linalg.matrix4_perspective(
			linalg.to_radians(f32(45.0)),
			f32(swapchain.extent.width) / f32(swapchain.extent.height),
			0.1,
			10.0,
		),
	}

	ubo.proj[1][1] *= -1

	mem.copy(uniform_buffers_mapped[current_frame], &ubo, size_of(ubo))
}

vk_must :: #force_inline proc(res: vk.Result, loc := #caller_location) {
	if res != .SUCCESS do log.panicf("vk: %v -", res, loc)
}

byte_arr_str :: proc(arr: ^[$N]byte) -> string {
	return strings.truncate_to_byte(string(arr[:]), 0)
}

main :: proc() {
	context.logger = log.create_console_logger(.Debug when ODIN_DEBUG else .Error)
	context.user_ptr = &Context{}

	using ctx := (^Context)(context.user_ptr)

	window_init()
	defer window_deinit()

	vk_init()
	defer vk_deinit()

	current_frame = 0

	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()

		vk_must(vk.WaitForFences(device, 1, &in_flight_fences[current_frame], true, max(u64)))

		img_idx: u32
		res := vk.AcquireNextImageKHR(
			device,
			swapchain.handle,
			max(u64),
			image_avail_sems[current_frame],
			{},
			&img_idx,
		)
		if res == .ERROR_OUT_OF_DATE_KHR || res == .SUBOPTIMAL_KHR || framebuffer_resized {
			framebuffer_resized = false
			vk_recreate_swapchain()
			continue
		} else {
			vk_must(res)
		}

		vk_must(vk.ResetFences(device, 1, &in_flight_fences[current_frame]))
		vk_must(vk.ResetCommandBuffer(command_buffers[current_frame], {}))

		vk_update_uniform_buffer()
		vk_record_command_buffer(command_buffers[current_frame], img_idx)

		submit_info := vk.SubmitInfo {
			sType                = .SUBMIT_INFO,
			waitSemaphoreCount   = 1,
			pWaitSemaphores      = &image_avail_sems[current_frame],
			pWaitDstStageMask    = &vk.PipelineStageFlags{.COLOR_ATTACHMENT_OUTPUT},
			commandBufferCount   = 1,
			pCommandBuffers      = &command_buffers[current_frame],
			signalSemaphoreCount = 1,
			pSignalSemaphores    = &render_finished_sems[current_frame],
		}

		vk_must(
			vk.QueueSubmit(queues[.Graphics], 1, &submit_info, in_flight_fences[current_frame]),
		)

		present_info := vk.PresentInfoKHR {
			sType              = .PRESENT_INFO_KHR,
			waitSemaphoreCount = 1,
			pWaitSemaphores    = &render_finished_sems[current_frame],
			swapchainCount     = 1,
			pSwapchains        = &swapchain.handle,
			pImageIndices      = &img_idx,
		}

		vk_must(vk.QueuePresentKHR(queues[.Present], &present_info))
		current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT
	}

	vk_must(vk.DeviceWaitIdle(device))
}
