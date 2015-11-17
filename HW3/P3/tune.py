import pyopencl as cl
import numpy as np

def create_data(N):
    return host_x, x

if __name__ == "__main__":
    N = 1e7

    platforms = cl.get_platforms()
    devices = [d for platform in platforms for d in platform.get_devices()]
    for i, d in enumerate(devices):
        print("#{0}: {1} on {2}".format(i, d.name, d.platform.name))
    ctx = cl.Context(devices)

    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    program = cl.Program(ctx, open('sum.cl').read()).build(options='')

    host_x = np.random.rand(N).astype(np.float32)
    x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_x)

    times = {}

    for num_workgroups in 2 ** np.arange(3, 10):
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in 2 ** np.arange(2, 8):
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_coalesced(queue, (num_workgroups * num_workers,), (num_workers,),
                                          x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4
            times['coalesced', num_workgroups, num_workers] = seconds
            print("coalesced reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    for num_workgroups in 2 ** np.arange(3, 10):
        partial_sums = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * num_workgroups)
        host_partial = np.empty(num_workgroups).astype(np.float32)
        for num_workers in 2 ** np.arange(2, 8):
            local = cl.LocalMemory(num_workers * 4)
            event = program.sum_blocked(queue, (num_workgroups * num_workers,), (num_workers,),
                                        x, partial_sums, local, np.uint64(N))
            cl.enqueue_copy(queue, host_partial, partial_sums, is_blocking=True)

            sum_gpu = sum(host_partial)
            sum_host = sum(host_x)
            seconds = (event.profile.end - event.profile.start) / 1e9
            assert abs((sum_gpu - sum_host) / max(sum_gpu, sum_host)) < 1e-4
            times['blocked', num_workgroups, num_workers] = seconds
            print("blocked reads, workgroups: {}, num_workers: {}, {} seconds".
                  format(num_workgroups, num_workers, seconds))

    best_time = min(times.values())
    best_configuration = [config for config in times if times[config] == best_time]
    print("configuration {}: {} seconds".format(best_configuration[0], best_time))