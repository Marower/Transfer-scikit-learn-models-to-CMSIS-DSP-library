################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code\ -\ example/CMSIS-DSP\ Library/Source/BasicMathFunctions/BasicMathFunctions.c \
C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code\ -\ example/CMSIS-DSP\ Library/Source/BasicMathFunctions/BasicMathFunctionsF16.c 

OBJS += \
./Source/BasicMathFunctions/BasicMathFunctions.o \
./Source/BasicMathFunctions/BasicMathFunctionsF16.o 

C_DEPS += \
./Source/BasicMathFunctions/BasicMathFunctions.d \
./Source/BasicMathFunctions/BasicMathFunctionsF16.d 


# Each subdirectory must supply rules for building sources it contributes
Source/BasicMathFunctions/BasicMathFunctions.o: C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code\ -\ example/CMSIS-DSP\ Library/Source/BasicMathFunctions/BasicMathFunctions.c Source/BasicMathFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32WB55xx -c -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/Include" -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/PrivateInclude" -I../USB_Device/App -I../USB_Device/Target -I../Core/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/STM32_USB_Device_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -I../Drivers/CMSIS/Device/ST/STM32WBxx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Source/BasicMathFunctions/BasicMathFunctionsF16.o: C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code\ -\ example/CMSIS-DSP\ Library/Source/BasicMathFunctions/BasicMathFunctionsF16.c Source/BasicMathFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32WB55xx -c -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/Include" -I"C:/Users/mzylinsk/STM32CubeIDE/workspace_1.10.1/PanTompkins/Code - example/CMSIS-DSP Library/PrivateInclude" -I../USB_Device/App -I../USB_Device/Target -I../Core/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc -I../Drivers/STM32WBxx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/STM32_USB_Device_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -I../Drivers/CMSIS/Device/ST/STM32WBxx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Source-2f-BasicMathFunctions

clean-Source-2f-BasicMathFunctions:
	-$(RM) ./Source/BasicMathFunctions/BasicMathFunctions.cyclo ./Source/BasicMathFunctions/BasicMathFunctions.d ./Source/BasicMathFunctions/BasicMathFunctions.o ./Source/BasicMathFunctions/BasicMathFunctions.su ./Source/BasicMathFunctions/BasicMathFunctionsF16.cyclo ./Source/BasicMathFunctions/BasicMathFunctionsF16.d ./Source/BasicMathFunctions/BasicMathFunctionsF16.o ./Source/BasicMathFunctions/BasicMathFunctionsF16.su

.PHONY: clean-Source-2f-BasicMathFunctions

