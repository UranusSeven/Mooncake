import argparse
import time
import statistics
import logging
import ctypes
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os
from mooncake.store import MooncakeDistributedStore
import threading
import queue
import copy
import math
import sys
import socket
import pickle
import struct
import uuid
from collections import defaultdict
from multiprocessing import Process, Queue as MPQueue

# Disable memcpy optimization
os.environ["MC_STORE_MEMCPY"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('stress_cluster_benchmark')


class KeyQueueServer:
    """TCP-based queue server for cross-server key distribution.

    This server manages a queue that prefill instances push keys to
    and decode instances pop keys from.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.queue = MPQueue()
        self.server_socket = None
        self.running = True

    def start(self):
        """Start the queue server (blocking)."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        logger.info(f"KeyQueueServer listening on {self.host}:{self.port}")

        try:
            while self.running:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"New client connection from {addr}")
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.server_socket.close()

    def stop(self):
        """Stop the queue server."""
        self.running = False

    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle a single client connection."""
        try:
            while self.running:
                # Read message length (4 bytes, big-endian)
                length_data = self._recv_exact(client_socket, 4)
                if not length_data:
                    break
                msg_length = struct.unpack('>I', length_data)[0]

                # Read message body
                data = self._recv_exact(client_socket, msg_length)
                if not data:
                    break

                request = pickle.loads(data)
                response = self._process_request(request)

                # Send response
                response_data = pickle.dumps(response)
                client_socket.sendall(struct.pack('>I', len(response_data)) + response_data)
        except ConnectionResetError:
            logger.info(f"Client {addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
        return data

    def _process_request(self, request: dict) -> dict:
        """Process a client request."""
        cmd = request.get('cmd')

        if cmd == 'push':
            keys = request.get('keys', [])
            for key in keys:
                self.queue.put(key)
            return {'status': 'ok', 'count': len(keys)}

        elif cmd == 'pop':
            count = request.get('count', 1)
            timeout = request.get('timeout', 10.0)
            keys = []
            for _ in range(count):
                try:
                    key = self.queue.get(timeout=timeout)
                    keys.append(key)
                except Exception:
                    break
            return {'status': 'ok', 'keys': keys}

        elif cmd == 'size':
            try:
                size = self.queue.qsize()
            except NotImplementedError:
                size = -1  # qsize() not implemented on some platforms
            return {'status': 'ok', 'size': size}

        elif cmd == 'ping':
            return {'status': 'ok', 'message': 'pong'}

        else:
            return {'status': 'error', 'message': f'Unknown command: {cmd}'}


class KeyQueueClient:
    """Client for connecting to KeyQueueServer across servers."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.lock = threading.Lock()
        self.connected = False

    def connect(self, retry_count: int = 5, retry_delay: float = 2.0):
        """Connect to the queue server with retries."""
        for attempt in range(retry_count):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                logger.info(f"Connected to KeyQueueServer at {self.host}:{self.port}")
                return
            except ConnectionRefusedError:
                if attempt < retry_count - 1:
                    logger.warning(f"Connection refused, retrying in {retry_delay}s... ({attempt + 1}/{retry_count})")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to KeyQueueServer at {self.host}:{self.port} after {retry_count} attempts")

    def _send_request(self, request: dict) -> dict:
        """Send a request and receive response."""
        if not self.connected:
            raise ConnectionError("Not connected to KeyQueueServer")

        with self.lock:
            data = pickle.dumps(request)
            self.socket.sendall(struct.pack('>I', len(data)) + data)

            # Read response length
            length_data = self._recv_exact(4)
            if not length_data:
                raise ConnectionError("Connection closed while reading response length")
            msg_length = struct.unpack('>I', length_data)[0]

            # Read response body
            response_data = self._recv_exact(msg_length)
            if not response_data:
                raise ConnectionError("Connection closed while reading response")

            return pickle.loads(response_data)

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def push_keys(self, keys: List[str]) -> int:
        """Push keys to the queue. Returns number of keys pushed."""
        response = self._send_request({'cmd': 'push', 'keys': keys})
        if response.get('status') != 'ok':
            raise RuntimeError(f"Push failed: {response.get('message', 'Unknown error')}")
        return response.get('count', 0)

    def pop_keys(self, count: int, timeout: float = 30.0) -> List[str]:
        """Pop keys from the queue. Returns list of keys."""
        response = self._send_request({'cmd': 'pop', 'count': count, 'timeout': timeout})
        if response.get('status') != 'ok':
            raise RuntimeError(f"Pop failed: {response.get('message', 'Unknown error')}")
        return response.get('keys', [])

    def get_size(self) -> int:
        """Get the current queue size."""
        response = self._send_request({'cmd': 'size'})
        return response.get('size', -1)

    def ping(self) -> bool:
        """Check if server is responsive."""
        try:
            response = self._send_request({'cmd': 'ping'})
            return response.get('status') == 'ok'
        except Exception:
            return False

    def close(self):
        """Close the connection."""
        if self.socket:
            self.socket.close()
            self.connected = False


def generate_random_key(prefix: str = "key") -> str:
    """Generate a random unique key using UUID."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class BatchResult:
    """Encapsulates the results of a batch operation with error handling."""
    
    keys: List[str]
    return_codes: List[int]
    operation_type: str
    
    def num_succeeded(self) -> int:
        """Return the number of successful operations."""
        if self.operation_type == "prefill":
            # For put operations: 0 = success, non-zero = error
            return sum(1 for code in self.return_codes if code == 0)
        else:
            # For get operations: positive = success (bytes read), negative = error
            return sum(1 for code in self.return_codes if code > 0)
    
    def num_failed(self) -> int:
        """Return the number of failed operations."""
        return len(self.return_codes) - self.num_succeeded()
    
    def get_failed_keys_with_codes(self) -> List[tuple[str, int]]:
        """Return a list of (key, error_code) tuples for failed operations."""
        failed = []
        for i, code in enumerate(self.return_codes):
            is_failed = (code != 0) if self.operation_type == "prefill" else (code < 0)
            if is_failed:
                failed.append((self.keys[i], code))
        return failed
    
    def log_failures(self, max_failures_to_log: int = 10):
        """Log detailed information about failed operations."""
        failed_ops = self.get_failed_keys_with_codes()
        if not failed_ops:
            return
        
        logger.warning(f"Batch {self.operation_type} had {len(failed_ops)} failures:")
        for i, (key, error_code) in enumerate(failed_ops[:max_failures_to_log]):
            logger.warning(f"  {key}: error_code={error_code}")
        
        if len(failed_ops) > max_failures_to_log:
            logger.warning(f"  ... and {len(failed_ops) - max_failures_to_log} more failures")


class PerformanceTracker:
    """Tracks and calculates performance metrics for operations."""

    def __init__(self):
        self.operation_latencies: List[float] = []
        self.operation_sizes: List[int] = []
        self.error_codes: Dict[int, int] = defaultdict(int)
        self.start_time: float = sys.float_info.max
        self.end_time: float = sys.float_info.min
        self.total_operations: int = 0
        self.failed_operations: int = 0
        self.bytes_transferred: int = 0

    def record_operation(self, latency_seconds: float, data_size_bytes: int):
        """Record a single operation's performance."""
        self.operation_latencies.append(latency_seconds)
        self.operation_sizes.append(data_size_bytes)
        
    def record_error(self, error_code: int):
        """Record an error code."""
        self.error_codes[error_code] += 1
        self.failed_operations += 1
        
    def extend(self, other: 'PerformanceTracker'):
        """Combine data from another tracker."""
        self.operation_latencies.extend(other.operation_latencies)
        self.operation_sizes.extend(other.operation_sizes)
        self.total_operations += other.total_operations
        self.failed_operations += other.failed_operations
        self.bytes_transferred += other.bytes_transferred
        self.start_time = min(self.start_time, other.start_time)
        self.end_time = max(self.end_time, other.end_time)
        
        # Merge error codes
        for code, count in other.error_codes.items():
            self.error_codes[code] += count
            
    def start_timer(self):
        """Start the overall timer for the test."""
        self.start_time = time.perf_counter()
        
    def stop_timer(self):
        """Stop the overall timer for the test."""
        self.end_time = time.perf_counter()
        
    def get_total_time(self) -> float:
        """Get the total wall time for the test."""
        return self.end_time - self.start_time if self.end_time > self.start_time else 0

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return comprehensive performance statistics."""
        if not self.operation_latencies:
            return {"error": "No operations recorded"}

        total_time = sum(self.operation_latencies)
        total_operations = len(self.operation_latencies)
        total_bytes = sum(self.operation_sizes)

        # Convert latencies to milliseconds for reporting
        latencies_ms = [lat * 1000 for lat in self.operation_latencies]

        # Calculate percentiles
        p90_latency = statistics.quantiles(latencies_ms, n=10)[8] if len(latencies_ms) >= 10 else max(latencies_ms)
        p99_latency = statistics.quantiles(latencies_ms, n=100)[98] if len(latencies_ms) >= 100 else max(latencies_ms)
        p999_latency = statistics.quantiles(latencies_ms, n=1000)[998] if len(latencies_ms) >= 1000 else max(latencies_ms)

        # Calculate throughput metrics
        ops_per_second = total_operations / total_time if total_time > 0 else 0
        bytes_per_second = total_bytes / total_time if total_time > 0 else 0
        mbps = bytes_per_second / (1024 * 1024)  # MB/s

        # Wall time throughput
        wall_time = self.get_total_time()
        wall_ops_per_second = total_operations / wall_time if wall_time > 0 else 0
        wall_mbps = total_bytes / wall_time / (1024 * 1024) if wall_time > 0 else 0

        return {
            "total_operations": total_operations,
            "succeeded_operations": total_operations - self.failed_operations,
            "failed_operations": self.failed_operations,
            "total_time_seconds": total_time,
            "wall_time_seconds": wall_time,
            "total_bytes": total_bytes,
            "p90_latency_ms": p90_latency,
            "p99_latency_ms": p99_latency,
            "p999_latency_ms": p999_latency,
            "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0,
            "min_latency_ms": min(latencies_ms) if latencies_ms else 0,
            "max_latency_ms": max(latencies_ms) if latencies_ms else 0,
            "operations_per_second": ops_per_second,
            "wall_operations_per_second": wall_ops_per_second,
            "throughput_mbps": mbps,
            "wall_throughput_mbps": wall_mbps,
            "throughput_bytes_per_second": bytes_per_second,
            "error_codes": dict(self.error_codes)
        }


class TestInstance:
    def __init__(self, args):
        self.args = args
        self.store = None
        self.performance_tracker = PerformanceTracker()
        self.buffer_array = None
        self.buffer_ptr = None
        self.key_queue_client: Optional[KeyQueueClient] = None

    def setup(self):
        """Initialize the MooncakeDistributedStore and allocate registered memory."""

        self.store = MooncakeDistributedStore()
        self.performance_tracker.start_timer()

        # Setup store
        protocol = self.args.protocol
        device_name = self.args.device_name
        local_hostname = self.args.local_hostname
        metadata_server = self.args.metadata_server
        global_segment_size = self.args.global_segment_size * 1024 * 1024
        local_buffer_size = self.args.local_buffer_size * 1024 * 1024
        master_server_address = self.args.master_server

        logger.info(f"Setting up {self.args.role} instance with batch_size={self.args.batch_size}")
        logger.info(f"  Protocol: {protocol}, Device: {device_name}")
        logger.info(f"  Global segment: {global_segment_size // (1024*1024)} MB")
        logger.info(f"  Local buffer: {local_buffer_size // (1024*1024)} MB")

        retcode = self.store.setup(local_hostname, metadata_server, global_segment_size,
                                  local_buffer_size, protocol, device_name, master_server_address)
        if retcode:
            logger.error(f"Store setup failed with return code {retcode}")
            exit(1)

        # Allocate and register memory buffer using numpy for better performance
        buffer_size = self.args.batch_size * self.args.value_length
        self.buffer_array = np.zeros(buffer_size, dtype=np.uint8)
        self.buffer_ptr = self.buffer_array.ctypes.data

        retcode = self.store.register_buffer(self.buffer_ptr, buffer_size)
        if retcode:
            logger.error(f"Buffer registration failed with return code {retcode}")
            exit(1)

        logger.info(f"Allocated and registered {buffer_size // (1024*1024)} MB buffer for zero-copy operations")

        # Connect to key queue server if address is provided
        if self.args.key_queue_server:
            host, port = self.args.key_queue_server.split(':')
            self.key_queue_client = KeyQueueClient(host, int(port))
            self.key_queue_client.connect()
            logger.info(f"Connected to key queue server at {self.args.key_queue_server}")

        time.sleep(1)

    def teardown(self):
        """Clean up resources."""
        if self.key_queue_client:
            self.key_queue_client.close()
            logger.info("Closed key queue client connection")

    def _calculate_total_batches(self) -> int:
        """Calculate the total number of batches needed."""
        return (self.args.max_requests + self.args.batch_size - 1) // self.args.batch_size

    def _run_benchmark(self, operation_type: str, operation_func):
        """Generic benchmark runner for both prefill and decode operations."""
        logger.info(f"Starting {operation_type} operations: {self.args.max_requests} requests")
        logger.info(f"Batch size: {self.args.batch_size}, Value size: {self.args.value_length // (1024*1024)} MB")

        total_operations = 0
        total_failed_operations = 0
        total_batches = self._calculate_total_batches()
        
        # Initialize progress bar for batch tracking
        with tqdm(total=total_batches, 
                  desc=f"{operation_type.capitalize()} batches",
                  unit="batch",
                  postfix={"failed_ops": 0}) as pbar:
            
            while total_operations < self.args.max_requests:
                # Calculate batch size for this iteration
                remaining = self.args.max_requests - total_operations
                current_batch_size = min(self.args.batch_size, remaining)
                
                # Prepare batch data
                keys = [f"key{total_operations + i}" for i in range(current_batch_size)]
                
		 # Measure batch operation latency
                op_start = time.perf_counter()
                return_codes = operation_func(keys, current_batch_size)
                op_end = time.perf_counter()

                operation_latency = op_end - op_start
                
                # Process results using BatchResult
                batch_result = BatchResult(keys, return_codes, operation_type)
                
                # Log failures if any (but limit verbosity)
                if batch_result.num_failed() > 0:
                    batch_result.log_failures(max_failures_to_log=3)
                
                successful_ops = batch_result.num_succeeded()
                failed_ops = batch_result.num_failed()
                total_failed_operations += failed_ops
                self.performance_tracker.failed_operations += failed_ops
                self.performance_tracker.total_operations += current_batch_size
                
                # Record error codes
                for code in return_codes:
                    if (operation_type == "prefill" and code != 0) or (operation_type == "decode" and code < 0):
                        self.performance_tracker.record_error(code)

                if successful_ops > 0:
                    # Record successful operations
                    total_data_size = successful_ops * self.args.value_length
                    self.performance_tracker.record_operation(operation_latency, total_data_size)
                    self.performance_tracker.bytes_transferred += total_data_size

                total_operations += current_batch_size
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"failed_ops": total_failed_operations})

        self.performance_tracker.stop_timer()
        logger.info(f"{operation_type.capitalize()} phase completed. Failed operations: {total_failed_operations}")
        self._print_performance_stats(operation_type.upper())

        logger.info(f"Waiting {self.args.wait_time} seconds...")
        time.sleep(self.args.wait_time)

    def _run_benchmark_with_random_keys(self, operation_type: str, operation_func):
        """Benchmark runner for prefill that generates random keys and pushes to queue."""
        logger.info(f"Starting {operation_type} operations: {self.args.max_requests} requests")
        logger.info(f"Batch size: {self.args.batch_size}, Value size: {self.args.value_length // (1024*1024)} MB")
        if self.key_queue_client:
            logger.info("Keys will be pushed to cross-server queue")

        total_operations = 0
        total_failed_operations = 0
        total_batches = self._calculate_total_batches()

        with tqdm(total=total_batches,
                  desc=f"{operation_type.capitalize()} batches",
                  unit="batch",
                  postfix={"failed_ops": 0}) as pbar:

            while total_operations < self.args.max_requests:
                remaining = self.args.max_requests - total_operations
                current_batch_size = min(self.args.batch_size, remaining)

                # Generate random keys for this batch
                keys = [generate_random_key(f"key_{self.args.thread_id}") for _ in range(current_batch_size)]

                op_start = time.perf_counter()
                return_codes = operation_func(keys, current_batch_size)
                op_end = time.perf_counter()

                operation_latency = op_end - op_start

                batch_result = BatchResult(keys, return_codes, operation_type)

                if batch_result.num_failed() > 0:
                    batch_result.log_failures(max_failures_to_log=3)

                successful_ops = batch_result.num_succeeded()
                failed_ops = batch_result.num_failed()
                total_failed_operations += failed_ops
                self.performance_tracker.failed_operations += failed_ops
                self.performance_tracker.total_operations += current_batch_size

                for code in return_codes:
                    if code != 0:
                        self.performance_tracker.record_error(code)

                if successful_ops > 0:
                    total_data_size = successful_ops * self.args.value_length
                    self.performance_tracker.record_operation(operation_latency, total_data_size)
                    self.performance_tracker.bytes_transferred += total_data_size

                total_operations += current_batch_size

                pbar.update(1)
                pbar.set_postfix({"failed_ops": total_failed_operations})

        self.performance_tracker.stop_timer()
        logger.info(f"{operation_type.capitalize()} phase completed. Failed operations: {total_failed_operations}")
        self._print_performance_stats(operation_type.upper())

        logger.info(f"Waiting {self.args.wait_time} seconds...")
        time.sleep(self.args.wait_time)

    def _run_benchmark_with_queue_keys(self, operation_type: str, operation_func):
        """Benchmark runner for decode that pops keys from queue."""
        logger.info(f"Starting {operation_type} operations: {self.args.max_requests} requests")
        logger.info(f"Batch size: {self.args.batch_size}, Value size: {self.args.value_length // (1024*1024)} MB")

        if not self.key_queue_client:
            logger.warning("No key queue client configured, falling back to sequential keys")
            return self._run_benchmark(operation_type, operation_func)

        logger.info("Keys will be popped from cross-server queue")

        total_operations = 0
        total_failed_operations = 0
        total_batches = self._calculate_total_batches()
        empty_queue_count = 0
        max_empty_retries = 10

        with tqdm(total=total_batches,
                  desc=f"{operation_type.capitalize()} batches",
                  unit="batch",
                  postfix={"failed_ops": 0, "queue_waits": 0}) as pbar:

            while total_operations < self.args.max_requests:
                remaining = self.args.max_requests - total_operations
                current_batch_size = min(self.args.batch_size, remaining)

                # Pop keys from the queue
                keys = self.key_queue_client.pop_keys(current_batch_size, timeout=30.0)

                if not keys:
                    empty_queue_count += 1
                    pbar.set_postfix({"failed_ops": total_failed_operations, "queue_waits": empty_queue_count})
                    if empty_queue_count >= max_empty_retries:
                        logger.warning(f"Queue empty after {max_empty_retries} retries, stopping")
                        break
                    logger.info(f"Queue empty, waiting... (attempt {empty_queue_count}/{max_empty_retries})")
                    time.sleep(1.0)
                    continue

                empty_queue_count = 0  # Reset on successful pop
                actual_batch_size = len(keys)

                op_start = time.perf_counter()
                return_codes = operation_func(keys, actual_batch_size)
                op_end = time.perf_counter()

                operation_latency = op_end - op_start

                batch_result = BatchResult(keys, return_codes, operation_type)

                if batch_result.num_failed() > 0:
                    batch_result.log_failures(max_failures_to_log=3)

                successful_ops = batch_result.num_succeeded()
                failed_ops = batch_result.num_failed()
                total_failed_operations += failed_ops
                self.performance_tracker.failed_operations += failed_ops
                self.performance_tracker.total_operations += actual_batch_size

                for code in return_codes:
                    if code < 0:
                        self.performance_tracker.record_error(code)

                if successful_ops > 0:
                    total_data_size = successful_ops * self.args.value_length
                    self.performance_tracker.record_operation(operation_latency, total_data_size)
                    self.performance_tracker.bytes_transferred += total_data_size

                total_operations += actual_batch_size

                pbar.update(1)
                pbar.set_postfix({"failed_ops": total_failed_operations, "queue_waits": empty_queue_count})

        self.performance_tracker.stop_timer()
        logger.info(f"{operation_type.capitalize()} phase completed. Failed operations: {total_failed_operations}")
        self._print_performance_stats(operation_type.upper())

        logger.info(f"Waiting {self.args.wait_time} seconds...")
        time.sleep(self.args.wait_time)

    def prefill(self):
        """Execute prefill operations using zero-copy batch put."""
        def put_batch(keys: List[str], batch_size: int) -> List[int]:
            # Prepare data in the registered buffer using numpy operations
            for i in range(batch_size):
                start_idx = i * self.args.value_length
                end_idx = start_idx + self.args.value_length

                # Simple pattern: fill with hash of key for randomness
                pattern = hash(keys[i]) % 256
                self.buffer_array[start_idx:end_idx] = pattern

            # Prepare buffer pointers and sizes for batch operation
            buffer_ptrs = []
            sizes = []
            for i in range(batch_size):
                offset = i * self.args.value_length
                buffer_ptrs.append(self.buffer_ptr + offset)
                sizes.append(self.args.value_length)

            return_codes = self.store.batch_put_from(keys, buffer_ptrs, sizes)

            # Push successfully written keys to the queue
            if self.key_queue_client:
                successful_keys = [
                    keys[i] for i, code in enumerate(return_codes) if code == 0
                ]
                if successful_keys:
                    self.key_queue_client.push_keys(successful_keys)
                    logger.debug(f"Pushed {len(successful_keys)} keys to queue")

            return return_codes

        self._run_benchmark_with_random_keys("prefill", put_batch)

    def decode(self):
        """Execute decode operations using zero-copy batch get."""
        def get_batch(keys: List[str], batch_size: int) -> List[int]:
            # Prepare buffer pointers and sizes for batch operation
            buffer_ptrs = []
            sizes = []
            for i in range(batch_size):
                offset = i * self.args.value_length
                buffer_ptrs.append(self.buffer_ptr + offset)
                sizes.append(self.args.value_length)

            return self.store.batch_get_into(keys, buffer_ptrs, sizes)

        self._run_benchmark_with_queue_keys("decode", get_batch)

    def _print_performance_stats(self, operation_type: str):
        """Print comprehensive performance statistics in a structured format."""
        stats = self.performance_tracker.get_statistics()

        if "error" in stats:
            logger.info(f"No performance data available for {operation_type}: {stats['error']}")
            return

        # Calculate success rate
        success_rate = (stats["succeeded_operations"] / stats["total_operations"]) * 100
        
        # Format bytes to human-readable format
        def format_bytes(size):
            if size == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB")
            i = int(math.floor(math.log(size, 1024)))
            p = math.pow(1024, i)
            s = round(size / p, 2)
            return f"{s} {size_name[i]}"

        # Build the entire report as a single string
        report = f"\n=== {operation_type} PERFORMANCE STATISTICS ===\n"
        report += f"Total operations: {stats['total_operations']}\n"
        report += f"  Succeeded: {stats['succeeded_operations']} ({success_rate:.2f}%)\n"
        report += f"  Failed:    {stats['failed_operations']}\n"
        report += f"Total data transferred: {format_bytes(stats['total_bytes'])}\n"
        report += f"Total wall time: {stats['wall_time_seconds']:.2f} seconds\n"
        report += f"Total operation time: {stats['total_time_seconds']:.2f} seconds\n"
        report += "Latency metrics:\n"
        report += f"  Mean latency: {stats['mean_latency_ms']:.2f} ms\n"
        report += f"  Min latency:  {stats['min_latency_ms']:.2f} ms\n"
        report += f"  Max latency:  {stats['max_latency_ms']:.2f} ms\n"
        report += f"  P90 latency:  {stats['p90_latency_ms']:.2f} ms\n"
        report += f"  P99 latency:  {stats['p99_latency_ms']:.2f} ms\n"
        report += f"  P999 latency: {stats['p999_latency_ms']:.2f} ms\n"
        report += "Throughput metrics:\n"
        report += f"  Operations/sec (operation time): {stats['operations_per_second']:.2f}\n"
        report += f"  Operations/sec (wall time):      {stats['wall_operations_per_second']:.2f}\n"
        report += f"  Throughput (operation time):     {stats['throughput_mbps']:.2f} MB/s\n"
        report += f"  Throughput (wall time):          {stats['wall_throughput_mbps']:.2f} MB/s\n"
        
        # Add error codes if any
        if stats['error_codes']:
            report += "Error codes encountered:\n"
            for code, count in stats['error_codes'].items():
                report += f"  Code {code}: {count} times\n"
                
        report += "===============================================\n"
        
        # Log the entire report as a single message
        logger.info(report)


def worker_thread(args, results_queue, start_barrier, end_barrier):
    """Worker thread function for executing tests."""
    try:
        thread_name = threading.current_thread().name
        logger.info(f"Worker thread {thread_name} initializing...")
        
        # Create tester instance and setup
        tester = TestInstance(args)
        tester.setup()
        
        # Wait for all threads to be ready
        logger.info(f"Worker thread {thread_name} waiting at start barrier")
        start_barrier.wait()
        logger.info(f"Worker thread {thread_name} passed start barrier")
        
        # Execute test
        if args.role == "decode":
            tester.decode()
        else:
            tester.prefill()

        # Cleanup
        tester.teardown()

        # Put results in the queue
        results_queue.put(tester.performance_tracker)
        logger.info(f"Worker thread {thread_name} completed successfully")
    except Exception as e:
        logger.error(f"Worker thread {threading.current_thread().name} failed: {e}")
        tracker = PerformanceTracker()
        tracker.record_error(-999)  # Special error code for thread failure
        results_queue.put(tracker)
    finally:
        # Ensure we always reach the end barrier
        logger.info(f"Worker thread {thread_name} waiting at end barrier")
        end_barrier.wait()
        logger.info(f"Worker thread {thread_name} passed end barrier")


def parse_arguments():
    """Parse command-line arguments for the stress test."""
    parser = argparse.ArgumentParser(
        description="Mooncake Distributed Store Zero-Copy Batch Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Role configuration
    parser.add_argument("--role", type=str, choices=["prefill", "decode"], default=None,
                       help="Role of this instance: prefill (producer) or decode (consumer). "
                            "Required unless running as queue server.")

    # Network and connection settings
    parser.add_argument("--protocol", type=str, default="rdma", help="Communication protocol to use")
    parser.add_argument("--device-name", type=str, default="erdma_0", help="Network device name for RDMA")
    parser.add_argument("--local-hostname", type=str, default="localhost", help="Local hostname")
    parser.add_argument("--metadata-server", type=str, default="http://127.0.0.1:8080/metadata", help="Metadata server address")
    parser.add_argument("--master-server", type=str, default="localhost:50051", help="Master server address")

    # Memory and storage settings
    parser.add_argument("--global-segment-size", type=int, default=10000, help="Global segment size in MB")
    parser.add_argument("--local-buffer-size", type=int, default=512, help="Local buffer size in MB")

    # Test parameters
    parser.add_argument("--max-requests", type=int, default=1200, help="Maximum number of requests to process")
    parser.add_argument("--value-length", type=int, default=4*1024*1024, help="Size of each value in bytes")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for operations")
    parser.add_argument("--wait-time", type=int, default=20, help="Wait time in seconds after operations complete")
    
    # Multi-threading parameters
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of worker threads to use for concurrent operations")
    
    # Statistics parameters
    parser.add_argument("--detailed-stats", action="store_true",
                       help="Enable detailed statistics per worker thread")

    # Cross-server key queue parameters
    parser.add_argument("--key-queue-server", type=str, default=None,
                       help="Address of the key queue server (host:port). "
                            "Prefill pushes keys, decode pops keys.")
    parser.add_argument("--run-queue-server", action="store_true",
                       help="Run as a key queue server instead of benchmark. "
                            "Use with --key-queue-server to specify bind address.")
    parser.add_argument("--queue-server-port", type=int, default=50052,
                       help="Port to bind when running as queue server")

    return parser.parse_args()


def print_performance_stats(stats: Dict[str, Any], title: str):
    """Print performance statistics in a structured format."""
    # Calculate success rate
    total_ops = stats["total_operations"]
    succeeded_ops = stats["succeeded_operations"]
    failed_ops = stats["failed_operations"]
    success_rate = (succeeded_ops / total_ops) * 100 if total_ops > 0 else 0
    
    # Format bytes to human-readable format
    def format_bytes(size):
        if size == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size, 1024)))
        p = math.pow(1024, i)
        s = round(size / p, 2)
        return f"{s} {size_name[i]}"

    # Build the entire report as a single string
    report = f"\n=== {title} PERFORMANCE STATISTICS ===\n"
    report += f"Total operations: {total_ops}\n"
    report += f"  Succeeded: {succeeded_ops} ({success_rate:.2f}%)\n"
    report += f"  Failed:    {failed_ops}\n"
    report += f"Total data transferred: {format_bytes(stats['total_bytes'])}\n"
    report += f"Total wall time: {stats['wall_time_seconds']:.2f} seconds\n"
    report += f"Total operation time: {stats['total_time_seconds']:.2f} seconds\n"
    report += "Latency metrics:\n"
    report += f"  Mean latency: {stats['mean_latency_ms']:.2f} ms\n"
    report += f"  Min latency:  {stats['min_latency_ms']:.2f} ms\n"
    report += f"  Max latency:  {stats['max_latency_ms']:.2f} ms\n"
    report += f"  P90 latency:  {stats['p90_latency_ms']:.2f} ms\n"
    report += f"  P99 latency:  {stats['p99_latency_ms']:.2f} ms\n"
    report += f"  P999 latency: {stats['p999_latency_ms']:.2f} ms\n"
    report += "Throughput metrics:\n"
    report += f"  Operations/sec (operation time): {stats['operations_per_second']:.2f}\n"
    report += f"  Operations/sec (wall time):      {stats['wall_operations_per_second']:.2f}\n"
    report += f"  Throughput (operation time):     {stats['throughput_mbps']:.2f} MB/s\n"
    report += f"  Throughput (wall time):          {stats['wall_throughput_mbps']:.2f} MB/s\n"
    
    # Add error codes if any
    if stats['error_codes']:
        report += "Error codes encountered:\n"
        for code, count in stats['error_codes'].items():
            report += f"  Code {code}: {count} times\n"
            
    report += "=" * 50 + "\n"
    
    # Log the entire report as a single message
    logger.info(report)


def run_queue_server(host: str, port: int):
    """Run the key queue server."""
    logger.info("=== Mooncake Key Queue Server ===")
    logger.info(f"Binding to {host}:{port}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 50)

    server = KeyQueueServer(host, port)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down queue server...")
        server.stop()


def main():
    """Main entry point for the stress test."""
    args = parse_arguments()

    # Handle queue server mode
    if args.run_queue_server:
        host = "0.0.0.0"
        port = args.queue_server_port
        if args.key_queue_server:
            parts = args.key_queue_server.split(':')
            host = parts[0] if parts[0] else "0.0.0.0"
            if len(parts) > 1:
                port = int(parts[1])
        run_queue_server(host, port)
        return

    # Validate role is provided for benchmark mode
    if not args.role:
        logger.error("--role is required when running benchmark (use --role prefill or --role decode)")
        sys.exit(1)

    logger.info("=== Mooncake Zero-Copy Batch Benchmark ===")
    logger.info(f"Role: {args.role.upper()}")
    logger.info(f"Protocol: {args.protocol}")
    logger.info(f"Max requests: {args.max_requests}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Value size: {args.value_length // (1024*1024)} MB")
    logger.info(f"Number of workers: {args.num_workers}")
    if args.key_queue_server:
        logger.info(f"Key queue server: {args.key_queue_server}")
    logger.info("=" * 50)

    try:
        if args.num_workers > 1:
            # Create barriers for precise timing
            start_barrier = threading.Barrier(args.num_workers + 1)  # +1 for main thread
            end_barrier = threading.Barrier(args.num_workers + 1)    # +1 for main thread
            
            # Multi-threaded execution
            results_queue = queue.Queue()
            threads = []
            
            # Adjust requests per worker
            requests_per_worker = args.max_requests // args.num_workers
            remainder = args.max_requests % args.num_workers
            
            # Create and start worker threads
            for i in range(args.num_workers):
                worker_args = copy.copy(args)
                worker_args.max_requests = requests_per_worker + (1 if i < remainder else 0)
                worker_args.thread_id = i + 1

                thread = threading.Thread(
                    target=worker_thread,
                    args=(worker_args, results_queue, start_barrier, end_barrier),
                    name=f"Worker-{i+1}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all workers to be ready at start barrier
            logger.info("Main thread waiting at start barrier")
            start_barrier.wait()
            logger.info("Main thread passed start barrier")
            
            # Wait for all workers to complete at end barrier
            logger.info("Main thread waiting at end barrier")
            end_barrier.wait()
            logger.info("Main thread passed end barrier")
            
            # Combine results
            combined_tracker = PerformanceTracker()
            worker_stats = []
            
            while not results_queue.empty():
                tracker = results_queue.get()
                worker_stats.append(tracker)
                combined_tracker.extend(tracker)

            # Print detailed worker stats if requested
            if args.detailed_stats:
                for i, tracker in enumerate(worker_stats):
                    stats = tracker.get_statistics()
                    print_performance_stats(stats, f"Worker {i+1} {args.role}")

            wall_time = combined_tracker.get_total_time()
            # Print combined statistics
            combined_stats = combined_tracker.get_statistics()
            print_performance_stats(combined_stats, "COMBINED PERFORMANCE")
            
            # Log precise wall time measurement
            logger.info(f"Precise wall time measurement: {wall_time:.4f} seconds")
        else:
            args.thread_id = 1
            # Single-threaded execution
            tester = TestInstance(args)
            tester.setup()

            if args.role == "decode":
                tester.decode()
            else:
                tester.prefill()

            tester.teardown()

        logger.info("Test completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
