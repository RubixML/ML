<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/ReLU6/ReLU6.php">[source]</a></span>

# ReLU6
ReLU6 is a variant of the standard Rectified Linear Unit (ReLU) that caps the maximum output value at 6. This bounded ReLU function is commonly used in mobile and quantized neural networks, where restricting the activation range can improve numerical stability and quantization efficiency.

$$
{\displaystyle ReLU6 = {\begin{aligned}&{\begin{cases}0&{\text{if }}x\leq 0\\x&{\text{if }}0 < x < 6\\6&{\text{if }}x \geq 6\end{cases}}=&\min\{6, \max\{0,x\}\}\end{aligned}}}
$$

## Parameters
This activation function does not have any parameters.

## Size and Performance
ReLU6 maintains the computational efficiency of standard ReLU while adding an upper bound check. It requires only simple comparison operations and conditional assignments. The additional upper bound check adds minimal computational overhead compared to standard ReLU, while providing benefits for quantization and numerical stability. This makes ReLU6 particularly well-suited for mobile and embedded applications where model size and computational efficiency are critical.

## Plots
<img src="../../images/activation-functions/relu6.png" alt="ReLU6 Function" width="500" height="auto">

<img src="../../images/activation-functions/relu6-derivative.png" alt="ReLU6 Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU6;

$activationFunction = new ReLU6();
```

