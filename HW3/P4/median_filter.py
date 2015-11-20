from __future__ import division
import pyopencl as cl
import numpy as np
import pylab
import os.path

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.pad(image, 1, mode='edge')
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        image = np.median(stacked, axis=2)

    return image

if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    curdir = os.path.dirname(os.path.realpath(__file__))
    program = cl.Program(context, open('median_filter.cl').read()).build(options=['-I', curdir])

    host_image = np.load('image.npz')['image'].astype(np.float32)[::2, ::2].copy()
    host_image_filtered = np.zeros_like(host_image)

    gpu_image_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)])
    width = np.int32(host_image.shape[1])
    height = np.int32(host_image.shape[0])

    # Set up a (N+2 x N+2) local memory buffer.
    # +2 for 1-pixel halo on all sides, 4 bytes for float.
    local_memory = cl.LocalMemory(4 * (local_size[0] + 2) * (local_size[1] + 2))
    # Each work group will have its own private buffer.
    buf_width = np.int32(local_size[0] + 2)
    buf_height = np.int32(local_size[1] + 2)
    halo = np.int32(1)

    # Send image to the device, non-blocking
    cl.enqueue_copy(queue, gpu_image_a, host_image, is_blocking=False)

    num_iters = 10
    for iter in range(num_iters):
        program.median_3x3(queue, global_size, local_size,
                           gpu_image_a, gpu_image_b, local_memory,
                           width, height,
                           buf_width, buf_height, halo)

        # swap filtering direction
        gpu_image_a, gpu_image_b = gpu_image_b, gpu_image_a

    cl.enqueue_copy(queue, host_image_filtered, gpu_image_a, is_blocking=True)

    assert np.allclose(host_image_filtered, numpy_median(host_image, num_iters))
