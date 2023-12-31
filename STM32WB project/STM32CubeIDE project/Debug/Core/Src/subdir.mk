################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/SVMLinearClassifier.c \
../Core/Src/SVMPolynomialClassifier.c \
../Core/Src/SVMSigmoidClassifier.c \
../Core/Src/SVMrbfClassifier.c \
../Core/Src/main.c \
../Core/Src/naiveBayesClassifier.c \
../Core/Src/stm32wbxx_hal_msp.c \
../Core/Src/stm32wbxx_it.c \
../Core/Src/syscalls.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32wbxx.c 

OBJS += \
./Core/Src/SVMLinearClassifier.o \
./Core/Src/SVMPolynomialClassifier.o \
./Core/Src/SVMSigmoidClassifier.o \
./Core/Src/SVMrbfClassifier.o \
./Core/Src/main.o \
./Core/Src/naiveBayesClassifier.o \
./Core/Src/stm32wbxx_hal_msp.o \
./Core/Src/stm32wbxx_it.o \
./Core/Src/syscalls.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32wbxx.o 

C_DEPS += \
./Core/Src/SVMLinearClassifier.d \
./Core/Src/SVMPolynomialClassifier.d \
./Core/Src/SVMSigmoidClassifier.d \
./Core/Src/SVMrbfClassifier.d \
./Core/Src/main.d \
./Core/Src/naiveBayesClassifier.d \
./Core/Src/stm32wbxx_hal_msp.d \
./Core/Src/stm32wbxx_it.d \
./Core/Src/syscalls.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32wbxx.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32WB55xx -c -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/Include" -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/PrivateInclude" -I../USB_Device/App -I../USB_Device/Target -I../Core/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/STM32_USB_Device_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -I../Drivers/CMSIS/Device/ST/STM32WBxx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/SVMLinearClassifier.cyclo ./Core/Src/SVMLinearClassifier.d ./Core/Src/SVMLinearClassifier.o ./Core/Src/SVMLinearClassifier.su ./Core/Src/SVMPolynomialClassifier.cyclo ./Core/Src/SVMPolynomialClassifier.d ./Core/Src/SVMPolynomialClassifier.o ./Core/Src/SVMPolynomialClassifier.su ./Core/Src/SVMSigmoidClassifier.cyclo ./Core/Src/SVMSigmoidClassifier.d ./Core/Src/SVMSigmoidClassifier.o ./Core/Src/SVMSigmoidClassifier.su ./Core/Src/SVMrbfClassifier.cyclo ./Core/Src/SVMrbfClassifier.d ./Core/Src/SVMrbfClassifier.o ./Core/Src/SVMrbfClassifier.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/naiveBayesClassifier.cyclo ./Core/Src/naiveBayesClassifier.d ./Core/Src/naiveBayesClassifier.o ./Core/Src/naiveBayesClassifier.su ./Core/Src/stm32wbxx_hal_msp.cyclo ./Core/Src/stm32wbxx_hal_msp.d ./Core/Src/stm32wbxx_hal_msp.o ./Core/Src/stm32wbxx_hal_msp.su ./Core/Src/stm32wbxx_it.cyclo ./Core/Src/stm32wbxx_it.d ./Core/Src/stm32wbxx_it.o ./Core/Src/stm32wbxx_it.su ./Core/Src/syscalls.cyclo ./Core/Src/syscalls.d ./Core/Src/syscalls.o ./Core/Src/syscalls.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32wbxx.cyclo ./Core/Src/system_stm32wbxx.d ./Core/Src/system_stm32wbxx.o ./Core/Src/system_stm32wbxx.su

.PHONY: clean-Core-2f-Src

