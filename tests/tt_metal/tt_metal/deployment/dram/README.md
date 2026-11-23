# DRAM Deployment Test Suite

This test suite finds DRAM faults on Blackhole chips before deployment. Each test writes data patterns to the DRAM banks of every chip. Then the test reads the data back and compares it with the expected values.

The tests run on all chips at the same time, one host thread for each chip.

## How a test works

Each test makes a list of jobs for each worker core. A job writes and reads one DRAM region with one data pattern. Each worker core runs its jobs in a persistent loop, so the host does not start a new program for each job.

Each worker core uses three kernels:

- The BRISC kernel (`kernels/dram_base_kernel.cpp`) makes the expected data and controls the job queue.
- The NCRISC kernel (`kernels/dram_second_kernel.cpp`) does all of the DRAM writes and reads.
- The compute kernel (`kernels/dram_compare_compute_kernel.cpp`) compares the data that it reads with the expected data.

A job walks its DRAM region one chunk at a time. For each chunk, the NCRISC kernel writes the pattern to DRAM, then reads it back.

The host monitors each worker core during the test. If a worker core makes no progress for 10 seconds, the stall watchdog stops the test. The test then fails.

## Test cases

The gtest names start with `MeshDispatchFixture.DramDeployment_`. All tests use the same patterns, loop count, and transfer settings. The first three are different in how they assign worker cores to DRAM banks. The two data-retention tests leave the data in DRAM for a fixed time before or after they check it.

### DramDeployment_PersistentOptimalWorkersAllDramBanks

This test uses one worker core for each DRAM bank. The test picks the worker core that has the best NOC 0 path to its bank.

Each worker core tests the full test region of its bank. All banks run at the same time.

Use this test to measure the bandwidth of each bank with the least NOC contention.

### DramDeployment_PersistentAllWorkersSingleDramSequentialSweep

This test uses all worker cores of the chip on one DRAM bank at a time. When one bank is finished, the test moves to the next bank.

The test divides the test region of the bank into equal parts, one part for each worker core. Each part is aligned to 4 KiB. The last worker core also gets the bytes that remain.

Use this test to put the maximum load on a single bank. It is the slowest of the three tests because the banks run one after the other.

### DramDeployment_PersistentPartitionedWorkersAllDramBanks

This test divides all worker cores among the DRAM banks. Worker core `n` tests bank `n % <number of banks>`.

The test divides the test region of each bank among the worker cores of that bank, the same as the sweep test. All banks run at the same time.

Use this test to put load on all banks at the same time, with many worker cores on each bank.

### DramDeployment_DataRetentionWriteSleepRead

This test runs each pass of each pattern in two phases. The host starts the read phase only after all worker cores have finished the write phase. Thus, no worker core reads a bank before the write to that bank is complete. All banks run at the same time.

1. Write phase. The test uses the same worker cores as `DramDeployment_PersistentOptimalWorkersAllDramBanks`: one worker core for each DRAM bank. Each worker core writes the pattern to the full test region of its bank. The test does not read back during this step. Then each worker core sleeps for `DRAM_TEST_RETENTION_SLEEP_S` seconds. The worker core does not access DRAM during the sleep.
2. Read phase. The test uses all worker cores of the chip:
   - Stress reads. Worker core `n` reads bank `n % <number of banks>`, the same as `DramDeployment_PersistentPartitionedWorkersAllDramBanks`. Each worker core reads 4 contiguous chunks of `DRAM_TEST_CHUNK_BYTES` bytes again and again, `DRAM_TEST_RETENTION_NUM_READS_PER_CORE` times in total (chunk 0, 1, 2, 3, 0, 1, ...). Worker core `j` of a bank reads chunks `4j` to `4j + 3` of the region. The test compares each read with the expected data.
   - Full check. After its stress reads, the worker core of the write phase for each bank reads the full region and compares it with the expected data. Thus, the test checks each byte that it wrote.

Use this test to find cells that do not keep their data while DRAM is not accessed, and cells that lose their data when nearby rows are read many times.

### DramDeployment_DataRetentionWriteReadSleep

This test is the same as `DramDeployment_DataRetentionWriteSleepRead`, but the sleep comes after the read phase. In the write phase, the worker cores do not sleep. In the read phase, each worker core sleeps after its last read job. The next write phase starts only after all worker cores have finished their sleep.

### Process-split data-retention cases (test_runner.py)

In the two tests above, the test process keeps the device open during the sleep. `test_runner.py` also runs two process-split cases, in which no process has the device open during the sleep:

- `RetentionSplitWriteSleepRead`: a write process writes the region and stops. The runner sleeps for `DRAM_TEST_RETENTION_SLEEP_S` seconds. Then a new read process does the stress reads and the full check.
- `RetentionSplitWriteReadSleep`: one process writes, then reads (stress reads and full check), and stops. The runner sleeps for `DRAM_TEST_RETENTION_SLEEP_S` seconds before the next cycle. The runner does not sleep after the last cycle.

A cycle is one pass of one pattern in one loop, in the same order as the tests above. There is one cycle for each sleep of the tests above. Each process runs one phase of one cycle with the gtests `DramDeployment_RetentionPhaseWrite` and `DramDeployment_RetentionPhaseRead`. The runner selects the cycle with `DRAM_TEST_RETENTION_CYCLE`. Without this variable, these two gtests are skipped. The worker cores do not sleep in these gtests.

The runner shows one result for each case. If a process does not complete its tests (for example, after a watchdog exit), the runner stops that case and shows it as `FAILED`.

Each process opens and closes the device, so the process-split cases take longer than the tests above.

### Notes on the data-retention tests

- The worker core sleeps with a wall-clock wait loop (`riscv_wait`). It updates its heartbeat during the sleep, so the stall watchdog does not stop the test.
- `DRAM_TEST_RETENTION_SLEEP_S` is a minimum. The wait loop counts AICLK cycles, and the test converts seconds to cycles at the maximum AICLK. AICLK is lower while the worker cores sleep (about 800 MHz on Blackhole), so the real sleep is longer, about 1.7 times the value. The chip summary shows the average real time per sleep.
- There is one sleep for each pass of each pattern in each loop. The default patterns have 142 passes, so one loop with the default 15 s sleep takes more than 35 minutes (about 60 minutes with the lower idle AICLK). With `DRAM_TEST_FAST=1` there are 5 passes, so one loop takes more than 75 seconds (about 2 minutes). The stress reads and the start of the program for each phase add more time.
- The chip summary time includes all sleeps, and the start and stop of the program for each phase.
- The 4 chunks of all worker cores of a bank must fit in the test region: `4 × <worker cores for each bank> × DRAM_TEST_CHUNK_BYTES` must not be more than `DRAM_TEST_MBYTES`. If they do not fit, the test stops with an error.
- The tests do not show a bandwidth summary, because the write and the read of a region are in different jobs.
- The test region of each bank starts at the unreserved DRAM base of metal (about 5.6 MiB on Blackhole), aligned up to 4 KiB, not at offset 0. Metal writes to the bottom of each bank (for example, the DRAM barrier word at offset 0) when it opens and closes the device. If the region does not fit below the test limit, the test makes it smaller. The other tests start at offset 0.
- `test_runner.py` runs the process-split cases, not the two in-process tests. To run the in-process tests, start the binary with `--gtest_filter=*DataRetention*`.
- All other variables (`DRAM_TEST_MBYTES`, `DRAM_TEST_LOOPS`, `DRAM_TEST_FAST`, `DRAM_TEST_MAX_CHIPS`, chunk size, transfer length mode, access mode, and NOC mode) have the same effect as in the other tests. In random access mode, the write and the full check of a region use the same seed, so they visit the chunks in the same order. The access mode and the transfer length mode have no effect on the stress reads. Each stress read is one full chunk.

## Data patterns

By default, each test runs these 13 patterns in this order. The number in parentheses is the number of passes for that pattern. Each pass is a full write and read of the region.

- Counter (1)
- Checkerboard (2)
- Address (1)
- MarchingOnes (32)
- MarchingZeroes (32)
- MarchingOneBits (33)
- MarchingZeroBits (33)
- ToggleBits (2)
- Saturation (2)
- ReversibleRandom (1)
- Random (1)
- RandomXoshiro128pp (1)
- ByteWiseSsn (1)

When `DRAM_TEST_FAST=1`, the tests run only Checkerboard, Random, Counter, and Address.

The random patterns start from the seed `0x12345678`. The seed increases by 1 for each loop (`DRAM_TEST_LOOPS`).

## Run the tests

Run all commands from the root of the repository. Build the tests before you run them.

To run the tests and get a summary, use the Python runner:

```sh
python3 tests/tt_metal/tt_metal/deployment/dram/test_runner.py
```

The runner starts `build/test/tt_metal/unit_tests_deployment` with `--gtest_filter=*DramDeployment_*:-*DataRetention*:*RetentionPhase*`. Then it runs the [process-split data-retention cases](#process-split-data-retention-cases-test_runnerpy), with one process for each phase. It reads the log while the tests run. At the end, it shows these tables:

- A PASS or FAIL table for each test, with one row for each chip and one column for each bank.
- A bandwidth table for each test, with read and write GB/s for each bank.
- A GDDR EDC table, only when `DRAM_TEST_EDC_CHECK=1`.
- A summary of the status of each test.

The runner also writes the results to a JSON file. The runner has these options:

| Option | Description |
|---|---|
| `-q` | Do not show the test log while the tests run. |
| `-n` | Do not show the summary tables. |
| `-v` | Show all log types. Without this option, the runner sets `TT_LOGGER_TYPES=Test`. |
| `-i <file>` | Parse this log file. The runner does not start the tests. |
| `-o <file>` | Write the JSON results to this file. The default is `out.json`. |
| `-l <file>` | Write the log and the summary tables to this file. |
| `--no-retention` | Do not run the process-split data-retention cases. |
| `--retention-only` | Run only the process-split data-retention cases. |

To run one test directly, use a gtest filter:

```sh
./build/test/tt_metal/unit_tests_deployment \
    --gtest_filter=MeshDispatchFixture.DramDeployment_PersistentOptimalWorkersAllDramBanks
```

To stop a test before it is finished, press Ctrl+C one time. The test stops after the current work and shows the results that it has. The test then shows as skipped.

The script `../test_runner.sh` runs the full deployment suite (Ethernet, DRAM, PCIe, and DIDT). It sets `TT_MESH_GRAPH_DESC_PATH` for the DRAM tests.

## Environment variables

All variables are optional. Set them in the shell before you start the test or the runner.

### Test size and duration

| Variable | Default | Description |
|---|---|---|
| `DRAM_TEST_MBYTES` | `32` | Size in MiB of the test region in each DRAM bank. The test starts at offset 0 of each bank. Values larger than the bank size minus 2048 bytes are reduced to that limit. The value must be more than 0. |
| `DRAM_TEST_LOOPS` | `1` | Number of times to run the full pattern list. A value of 0, or a value that is not a number, gives 1. Each loop tests the same region again. |
| `DRAM_TEST_FAST` | off | Set to `1` to run only 4 patterns: Checkerboard, Random, Counter, and Address. |
| `DRAM_TEST_MAX_CHIPS` | all chips | Number of chips to test. The tests use the first chips in the device list. The value must be from 1 to the number of chips. |
| `DRAM_TEST_RETENTION_SLEEP_S` | `15` | Data-retention tests only. Minimum number of seconds to sleep between the write and the read (`WriteSleepRead`), or after the read (`WriteReadSleep`). A value of 0 gives no sleep. A value that is not a number gives 15. |
| `DRAM_TEST_RETENTION_NUM_READS_PER_CORE` | `20000` | Data-retention tests only. Number of stress reads that each worker core does in the read phase. Each read is one chunk of `DRAM_TEST_CHUNK_BYTES` bytes. A value of 0, or a value that is not a number, gives 20000. |

`DRAM_TEST_MBYTES` is the size of one pass on one bank. Each pass of each pattern writes and reads the full region. Each loop does all of the passes again, on the same region. A larger `DRAM_TEST_LOOPS` value does not test more DRAM addresses. It tests the same addresses more times, with a new seed.

For example, the default patterns have 142 passes in total. With `DRAM_TEST_MBYTES=32` and `DRAM_TEST_LOOPS=2`, each bank writes and reads 32 MiB × 142 × 2 = 9088 MiB. All of this data goes to the same 32 MiB of each bank.

### Transfer shape and access order

| Variable | Default | Description |
|---|---|---|
| `DRAM_TEST_CHUNK_BYTES` | `4096` | Size in bytes of each NOC transfer. The value must be a multiple of 4 and not more than 1 MiB. Use a multiple of 64 (one NOC word) to get exact chunk sizes. |
| `DRAM_TEST_TRANSFER_LEN_MODE` | `0` | How the test sets the size of each transfer. Refer to [Transfer length modes](#transfer-length-modes). |
| `DRAM_TEST_ACCESS_MODE` | `sequential` | The order in which a job visits the chunks of its region. Values are `0` or `sequential`, and `1` or `random`. Refer to [Access modes](#access-modes). |
| `DRAM_TEST_NOC_MODE` | `0` | Which NOC the test uses for DRAM writes and reads. Refer to [NOC modes](#noc-modes). |

#### Transfer length modes

These modes apply only to sequential access. Random access ignores `DRAM_TEST_TRANSFER_LEN_MODE`.

| Value | Name | Description |
|---|---|---|
| `0` | Fixed length | Each transfer is `DRAM_TEST_CHUNK_BYTES`. The last transfer of a region can be smaller. |
| `1` | Random length | Each transfer has a random size from 1 to `DRAM_TEST_CHUNK_BYTES`. The test rounds the size up to a multiple of 64 bytes. |
| `2` | Tile-like length | Each transfer has a random size like a tile: a random multiple of 512 bytes (0 to 15872), plus a random header of 0, 64, or 128 bytes. The maximum is `DRAM_TEST_CHUNK_BYTES`. The minimum is 64 bytes. |

`DRAM_TEST_CHUNK_BYTES` is always the maximum transfer size. The modes do not replace it.

In mode `2`, most random sizes are larger than 4096 bytes. As a result, with the default chunk size, most transfers are 4096 bytes. To get more different sizes, set `DRAM_TEST_CHUNK_BYTES` to a larger value, for example `16384`.

The random sizes come from the job seed and the pass number. Thus, a run with the same settings gives the same sizes.

#### Access modes

| Value | Name | Description |
|---|---|---|
| `0` or `sequential` | Sequential | A job visits the chunks in order of address: offset 0, then 1 chunk, then 2 chunks, and so on. This mode gives the best DRAM row locality. The transfer length mode applies. |
| `1` or `random` | Random | A job visits each chunk one time, in a shuffled order. The DRAM addresses jump across the region, so the test causes more row activations and page misses. The transfer length mode has no effect. |

Random mode divides the region into a fixed grid of chunks. The chunk size is `DRAM_TEST_CHUNK_BYTES`, rounded down to a multiple of 64 bytes. The minimum chunk size is 64 bytes. The last chunk is smaller when the region is not a multiple of the chunk size.

The shuffled order comes from the job seed and the pass number. Each pass uses a different order.

#### NOC modes

| Value | Description |
|---|---|
| `0` | All writes and reads use NOC 0. |
| `1` | All writes and reads use NOC 1. |
| `alternate` | Writes and reads use different NOCs. The NOCs change places for each pattern. For the first pattern, writes use NOC 0 and reads use NOC 1. |

### Reports and logs

| Variable | Default | Description |
|---|---|---|
| `DRAM_TEST_EDC_CHECK` | off | Set to `1` to read the GDDR EDC telemetry after all tests. Refer to [EDC check](#edc-check). |
| `DRAM_TEST_PATTERN_TIMING` | off | Set to `1` to show the time that each pattern used, in a table at the end of each test. |
| `DRAM_TEST_VERBOSE` | off | Set to `1` to show each job, each assignment of a worker core to a bank, and more. |

#### EDC check

When `DRAM_TEST_EDC_CHECK=1`, the suite reads the GDDR EDC counters from the firmware one time. It does this after the last test, for each chip that a test used. The suite shows a table with these values for each GDDR module:

- Top and bottom temperature.
- Corrected read and write errors. These counters stop at 255.
- Uncorrectable read and write error flags (0 or 1).

If an uncorrectable flag is set on a chip, the suite fails.

All counters and flags start at 0 when the chip resets. Thus, a flag can come from an earlier run on the same chip. Reset the chip before the test to make sure that the result is from this run only.

If the firmware does not give EDC telemetry, the check passes for that chip. If the process stops before the end of the suite, the EDC check does not run.

### Fault injection

Use these variables only to make sure that the test finds faults. Each one makes the test fail.

| Variable | Default | Description |
|---|---|---|
| `DRAM_INSERT_ERRORS_PATTERN_ID` | off | Set to a pattern ID to corrupt the data that the test reads back. The test corrupts the first 20 words of that pattern, one time, on DRAM bank 0, in the first pass of the first loop. The DRAM data does not change. |
| `DRAM_TEST_INJECT_TENSIX_HEARTBEAT_STALL` | off | Set to `1` to stop a worker core on purpose, after 10 transfers of job 1. The stall watchdog must then stop the test with the reason `kernel_stall`. |

The pattern IDs for `DRAM_INSERT_ERRORS_PATTERN_ID` are:

| ID | Pattern |
|---|---|
| 1 | Random |
| 2 | MarchingOnes |
| 3 | MarchingZeroes |
| 4 | ReversibleRandom |
| 5 | RandomXoshiro128pp |
| 6 | ToggleBits |
| 7 | Saturation |
| 8 | MarchingOneBits |
| 9 | MarchingZeroBits |
| 10 | Counter |
| 11 | Address |
| 12 | ByteWiseSsn |

Checkerboard has the ID 0. A value of 0 turns off error injection, so you cannot inject errors into Checkerboard.

CAUTION: Do not use `DRAM_TEST_INJECT_TENSIX_HEARTBEAT_STALL=1` on a production run. The worker core stays in the stall, and the process can stop before the device cleanup is complete.

If the stall watchdog stops the process, the device can stay in a bad state. If the next run cannot get the device, reset the chips:

```sh
source ~/.tenstorrent-venv/bin/activate
tt-smi -r
```

## Results

Each test shows these results at the end:

- A summary for each chip, with the number of jobs, the time, the bytes checked, and the pass status.
- A bank result table, with one row for each chip and `OK` or `NO` for each bank.
- A bandwidth table, with the read and write bandwidth of each bank in GB/s and bytes for each cycle.
- A Galaxy summary for all chips. If a test fails, this summary also shows the write and read error rates.

The bank table shows the PCI BDF, the UBB tray, and the location of each chip. Use these values to find a bad chip in a Galaxy system.

The test classifies each mismatch as a suspected write error or a suspected read error. A log line that starts with `Mismatch on dram_controller=` gives the bank, the worker core, the pattern, and the counts.

The bandwidth is the NCRISC transfer time only. It includes the wait for each NOC transfer to complete, but not the data generation or the compare.

To convert the JSON results of the runner to CSV, use `analyze_test_output.py`:

```sh
python3 tests/tt_metal/tt_metal/deployment/dram/analyze_test_output.py -i out.json -o out.csv
```
