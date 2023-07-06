/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include "usbd_cdc_if.h"
#include <math.h>
#include "arm_math.h"
//VECTOR_DIMENSION should match input vector length of classifier
#define VECTOR_DIMENSION 7

#include "naiveBayesClassifier.h"
#include "SVMLinearClassifier.h"
#include "SVMSigmoidClassifier.h"
#include "SVMrfbClassifier.h"
#include "SVMPolynomialClassifier.h"

#include "featuresDataset.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */


/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim16;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM16_Init(void);
static void MX_TIM2_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

unsigned char str[250];

uint8_t USBBuffer[2048];
uint16_t USBBuffer_index = 0;

void sendUSBData (uint8_t* Buf, uint16_t Len)
{
	memcpy(&USBBuffer[USBBuffer_index], Buf, Len);
	USBBuffer_index += Len;
}

uint16_t findStrEnd ()
{
	uint16_t end = 10;
	while (str[end]!='\0')
	{
		end++;
		if (end == 250)
		{
			break;
		}
	}
	return end;
}

void sendPrediction (float32_t innputVector[])
{
	uint32_t prediction = 0;
	prediction = predictClassNaiveBayes(innputVector);
	sprintf(str,"Naive Bayes: %d\r\n", prediction);
	sendUSBData (str, findStrEnd());

	prediction = predictSVMLinear (innputVector);
	sprintf(str,"SVM Linear: %d\r\n", prediction);
	sendUSBData (str, findStrEnd());

	prediction = predictSVMPolynomial (innputVector);
	sprintf(str,"SVM Polynomial: %d\r\n", prediction);
	sendUSBData (str, findStrEnd());

	prediction = predictSVMrbf (innputVector);
	sprintf(str,"SVM RBF: %d\r\n", prediction);
	sendUSBData (str, findStrEnd());

	prediction = predictSVMSigmoid (innputVector);
	sprintf(str,"SVM Sigmoid: %d\r\n", prediction);
	sendUSBData (str, findStrEnd());
}

void goThroughDataset ()
{
	 float32_t input[7];
	 for (uint32_t i = 0; i < leanghtOfFeaturesArray; i++)
	 {
		 memcpy(&input[0],&featuresArray[i],7*4);

		 sprintf(str,"%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n", i, input[0], input[1], input[2], input[3], input[4], input[5], input[6]);
		 sendUSBData (str, findStrEnd());


		 sendPrediction(input);
	 }
}

void testTimeOfComputation()
{   float32_t input[7];
	uint32_t computationTime;
	uint16_t count = 0;
	uint16_t N = 100;
	TIM2->CNT = 0;
	for (uint16_t i = 0; i < 10; i++)
	{
		count = 0;
		do
		{
			memcpy(&input[0],&featuresArray[count],7*4);
			count++;
			predictClassNaiveBayes(input);
		}while (count < leanghtOfFeaturesArray);
	}
	computationTime = TIM2->CNT;

	sprintf(str,"NaiveBayes: %u ticks\r\n", computationTime);
	sendUSBData (str, findStrEnd());

	TIM2->CNT = 0;
	for (uint16_t i = 0; i < 10; i++)
	{
		count = 0;
		do
		{
			memcpy(&input[0],&featuresArray[count],7*4);
			count++;
			predictSVMLinear(input);
		}while (count < leanghtOfFeaturesArray);
	}
	computationTime = TIM2->CNT;

	sprintf(str,"SVM linear: %u ticks\r\n", computationTime);
	sendUSBData (str, findStrEnd());

	TIM2->CNT = 0;
	for (uint16_t i = 0; i < 10; i++)
	{
		count = 0;
		do
		{
			memcpy(&input[0],&featuresArray[count],7*4);
			count++;
			predictSVMPolynomial(input);
		}while (count < leanghtOfFeaturesArray);
	}
	computationTime = TIM2->CNT;

	sprintf(str,"SVM polynomial: %u ticks\r\n", computationTime);
	sendUSBData (str, findStrEnd());

	TIM2->CNT = 0;
	for (uint16_t i = 0; i < 10; i++)
	{
		count = 0;
		do
		{
			memcpy(&input[0],&featuresArray[count],7*4);
			count++;
			predictSVMrbf(input);
		}while (count < leanghtOfFeaturesArray);
	}
	computationTime = TIM2->CNT;

	sprintf(str,"SVM rbf: %u ticks\r\n", computationTime);
	sendUSBData (str, findStrEnd());

	TIM2->CNT = 0;
	for (uint16_t i = 0; i < 10; i++)
	{
		count = 0;
		do
		{
			memcpy(&input[0],&featuresArray[count],7*4);
			count++;
			predictSVMSigmoid(input);
		}while (count < leanghtOfFeaturesArray);
	}
	computationTime = TIM2->CNT;

	sprintf(str,"SVM sigmoid: %u ticks\r\n", computationTime);
	sendUSBData (str, findStrEnd());
}

void parseLine (uint8_t* Buf, uint32_t Len)
{//Function parse one line from serial port
	if (Len < 7)
	{
		switch (Buf[0])
		{
			case 'A':
				goThroughDataset ();
				break;
			case 'B':
				testTimeOfComputation();
				break;
			default:
				sendUSBData (Buf, Len);
		}

	}
	else
	{
		float32_t in[VECTOR_DIMENSION];
		uint32_t i = 0;
		uint32_t index = 0;
		uint32_t count = 0;
		uint32_t inCount = 0;
		char word[20];
		do
		{
			if((inCount)==VECTOR_DIMENSION)
					break;
			if (Buf[i]==',' || Buf[i]=='\n')
			{
				if (count>19)
				{
					count = 19;
				}
				memcpy(&word[0],&Buf[index],count);
				word[count+1] = '\0';
				in[inCount] = (float)atof(word);
				inCount++;
				i++;
				index = i;
				count = 0;

			}
			i++;
			count++;
		}while (i<Len);
		if((inCount)==VECTOR_DIMENSION)
		{
			sendPrediction (in);
		}
		else
		{
			sprintf(str,"error, wrong input length.\r\n");
			sendUSBData (str,findStrEnd());
		}
	}


}

extern uint8_t bufferFlag;
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
	initARMGaussianNaiveBayesClasificator ();
	initARMSVMLinearClasificator ();
	initARMSVMPolynomialClasificator ();
	initARMSVMrbfClasificator ();
	initARMSVMSigmoidClasificator ();
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

/* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USB_Device_Init();
  MX_TIM16_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  extern uint32_t byteInBuffer;


  extern uint8_t RxBufferFS[APP_RX_DATA_SIZE];
  HAL_TIM_Base_Start_IT(&htim16);
  HAL_TIM_Base_Start_IT(&htim2);
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  if ((byteInBuffer>0) && (bufferFlag == 8))
	 	  {
	 		  uint32_t count = 0;
	 		  uint32_t i = 0;
	 		  uint32_t index = 0;
	 		  do
	 		  {
	 			  i++;
	 			  count++;
	 			  if (RxBufferFS[i]=='\n')
	 			  {
	 				 count++;
	 				 parseLine (&RxBufferFS[index], count);
	 				 index += count;
	 				 i++;
	 				 count = 0;
	 			  }
	 			  if (i >= byteInBuffer)
	 			  {
	 				  break;
	 			  }
	 		  }while (RxBufferFS[i] != '\0');
	 		  if (byteInBuffer != index)
	 		  {
				 memcpy(&RxBufferFS[0],&RxBufferFS[index],byteInBuffer-index);
				 byteInBuffer -=index;
	 		  }
	 		  else
	 		  {
	 			 byteInBuffer = 0;
	 		  }
	 		  bufferFlag = 0;
	 	  }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_LSE
                              |RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_OFF;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 32;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the SYSCLKSource, HCLK, PCLK1 and PCLK2 clocks dividers
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK4|RCC_CLOCKTYPE_HCLK2
                              |RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.AHBCLK2Divider = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.AHBCLK4Divider = RCC_SYSCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SMPS;
  PeriphClkInitStruct.SmpsClockSelection = RCC_SMPSCLKSOURCE_HSI;
  PeriphClkInitStruct.SmpsDivSelection = RCC_SMPSCLKDIV_RANGE1;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN Smps */

  /* USER CODE END Smps */
}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 63;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM16 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM16_Init(void)
{

  /* USER CODE BEGIN TIM16_Init 0 */

  /* USER CODE END TIM16_Init 0 */

  /* USER CODE BEGIN TIM16_Init 1 */

  /* USER CODE END TIM16_Init 1 */
  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 639;
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim16.Init.Period = 500;
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM16_Init 2 */

  /* USER CODE END TIM16_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();

}

/* USER CODE BEGIN 4 */
// Callback: timer has rolled over
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  // Check which version of the timer triggered this callback and toggle LED
  if (htim == &htim16)
  {
	  if (USBBuffer_index> 0)
	  {
		  while (CDC_Transmit_FS(USBBuffer, USBBuffer_index)!= USBD_OK){};
		  USBBuffer_index = 0;
	  }
	  if (bufferFlag<8)
	  {
		  bufferFlag++;
	  }
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
