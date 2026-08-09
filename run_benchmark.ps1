# Prevent Windows sleep while running benchmark
$es = Add-Type -MemberDefinition '[DllImport("kernel32.dll")]public static extern uint SetThreadExecutionState(uint esFlags);' -Name KeepAwake -Namespace Win32 -PassThru
$null = $es::SetThreadExecutionState(2147483651)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED

try {
    python references/video_classification/benchmark_inference.py --device cpu
} finally {
    $null = $es::SetThreadExecutionState(2147483648)  # ES_CONTINUOUS only (resets)
}
