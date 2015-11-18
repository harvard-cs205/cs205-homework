from __future__ import division
import pyopencl as cl
import numpy as np
import pylab

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


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
    devices = platforms[1].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('mandelbrot.cl').read()).build(options='')

    in_coords, out_counts = make_coords()
    real_coords = np.real(in_coords).copy()
    imag_coords = np.imag(in_coords).copy()

    gpu_real = cl.Buffer(context, cl.mem_flags.READ_ONLY, real_coords.size * 4)
    gpu_imag = cl.Buffer(context, cl.mem_flags.READ_ONLY, real_coords.size * 4)
    gpu_counts = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, real_coords.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(in_coords.shape[::-1], local_size)])
    width = np.int32(in_coords.shape[1])
    height = np.int32(in_coords.shape[0])
    max_iters = np.int32(1024)

    cl.enqueue_copy(queue, gpu_real, real_coords, is_blocking=False)
    cl.enqueue_copy(queue, gpu_imag, imag_coords, is_blocking=False)

    event = program.mandelbrot(queue, global_size, local_size,
                               gpu_real, gpu_imag, gpu_counts,
                               width, height, max_iters)

    cl.enqueue_copy(queue, out_counts, gpu_counts, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9
    print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))
    pylab.imshow(np.log(out_counts))
    pylab.show()
