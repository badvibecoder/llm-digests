# Deepseek Oneapi SYCL Testing

These files should go in your project directory. The container will look at the $PWD as its workspace.

## Docker Bypass

We need to  add 0666 perms so the agent can bounce easily off of docker.

The otherway would be to execute dsh from within docker, simply adding dsh/nodejs/bun to the oneapi contain is the next logical step.

```bash
#!/usr/bin/env bash

# Step 1: Create a systemd drop-in override directory for docker.socket
sudo mkdir -p /etc/systemd/system/docker.socket.d

# Step 2: Write the override configuration to make the socket world-writable (0666)
cat << 'EOF' | sudo tee /etc/systemd/system/docker.socket.d/override.conf
[Socket]
SocketMode=0666
EOF

# Step 3: Reload the systemd daemon to pick up the changes
sudo systemctl daemon-reload

# Step 4: Restart the Docker socket and service
sudo systemctl restart docker.socket docker.service

# Step 5: Verify the socket permissions are now srw-rw-rw-
ls -l /var/run/docker.sock
```

## `start-oneapi.sh`

This will create the container for long duration and allow dsh to interact with the container using the native host terminal.

Be sure to change the `CONTAINER_NAME` variable in the script. In this document when you see `oneapi-CHANGEME` it means you should be setting this to the same name for all scripts.

Keep in mind the workspace for the container will be whatever folder you execute this from.

```bash
#!/bin/bash
PROJECT_NAME="$(basename "$PWD")"
CONTAINER_NAME="oneapi-CHANGEME"

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

# Start if not already running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting background container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --device /dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        --group-add "$VIDEO_GID" \
        --group-add "$RENDER_GID" \
        --user "$(id -u):$(id -g)" \
        -v "$PWD:/workspace" \
        -w /workspace \
        intel/oneapi-toolkit:latest \
        sleep infinity
fi
```

Launch the container: `bash start-oneapi.sh`

## `cmd-oneapi.sh`

Make sure the previous `CONTAINER_NAME` value matches. This script will allow you to execute your command line against the project container.

```bash
#!/usr/bin/env bash

#!/bin/bash
PROJECT_NAME="$(basename "$PWD")"
CONTAINER_NAME="oneapi-CHANGEME"

# Forward any command directly to the container
docker exec -w /workspace "${CONTAINER_NAME}" "$@"
```

## Usage

Compile.

```
./cmd-oneapi.sh icpx -fsycl -O2 hello_sycl.cpp -o hello_sycl
```

Run.

```
./cmd-oneapi.sh ./hello_sycl
Device: Intel(R) Graphics [0xe223]
Driver: 1.14.37020+3
Kernel Output: 0 10 20 30 40 50 60 70 
Hello World from SYCL GPU kernel!
```

## Teardown

When done working we need to teardown this infinitely running container. Data will be saved in our workspace for the next launch

`stop-oneapi.sh`

```bash
#!/usr/bin/env bash

# Step 1: Force stop and remove the container immediately
docker rm -f oneapi-CHANGEME

# Step 2: Verify the container has been terminated
docker ps -a --filter "name=oneapi-CHANGEME"
```

## Hello World Test

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        // Enforce execution on the Intel GPU
        sycl::queue q(sycl::gpu_selector_v);

        std::cout << "Device: " 
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << "\nDriver: " 
                  << q.get_device().get_info<sycl::info::device::driver_version>() 
                  << std::endl;

        constexpr size_t N = 8;
        int* data = sycl::malloc_shared<int>(N, q);

        // Run a parallel kernel on the GPU
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            data[idx] = static_cast<int>(idx[0]) * 10;
        }).wait();

        std::cout << "Kernel Output: ";
        for (size_t i = 0; i < N; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\nHello World from SYCL GPU kernel!" << std::endl;

        sycl::free(data, q);
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception caught: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```