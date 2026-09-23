import triton

target = triton.runtime.driver.active.get_current_target()
print(target)