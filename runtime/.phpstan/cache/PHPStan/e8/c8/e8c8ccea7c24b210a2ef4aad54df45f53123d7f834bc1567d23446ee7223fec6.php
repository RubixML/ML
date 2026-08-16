<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/GELU.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\GELU
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-80af65a0a65ada1a2b969cd1a21844e1fed28840c114d3ea9ae1f5c849e7631d',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/GELU.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
    'shortName' => 'GELU',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * GeLU
 *
 * Gaussian Error Linear Units (GeLUs) are rectifiers that are gated by the magnitude of their input rather
 * than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value
 * of a neuron with random dropout regularization applied.
 *
 * References:
 * [1] D. Hendrycks et al. (2018). Gaussian Error Linear Units (GeLUs).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 29,
    'endLine' => 151,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\ActivationFunction',
      1 => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\IBufferDerivative',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'ALPHA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'name' => 'ALPHA',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.7978845608',
          'attributes' => 
          array (
            'startLine' => 36,
            'endLine' => 36,
            'startTokenPos' => 65,
            'startFilePos' => 1032,
            'endTokenPos' => 65,
            'endFilePos' => 1043,
          ),
        ),
        'docComment' => '/**
 * The square root of two over pi constant sqrt(2/π).
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 36,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
      'HALF_ALPHA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'name' => 'HALF_ALPHA',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.3989422804',
          'attributes' => 
          array (
            'startLine' => 41,
            'endLine' => 41,
            'startTokenPos' => 78,
            'startFilePos' => 1126,
            'endTokenPos' => 78,
            'endFilePos' => 1137,
          ),
        ),
        'docComment' => '/**
 * @var float 0.5 * ALPHA
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 41,
        'endLine' => 41,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
      'BETA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'name' => 'BETA',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.044715',
          'attributes' => 
          array (
            'startLine' => 48,
            'endLine' => 48,
            'startTokenPos' => 91,
            'startFilePos' => 1260,
            'endTokenPos' => 91,
            'endFilePos' => 1267,
          ),
        ),
        'docComment' => '/**
 * Gaussian error function approximation term.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 48,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 36,
      ),
      'TRIPLE_BETA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'name' => 'TRIPLE_BETA',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.134145',
          'attributes' => 
          array (
            'startLine' => 53,
            'endLine' => 53,
            'startTokenPos' => 104,
            'startFilePos' => 1348,
            'endTokenPos' => 104,
            'endFilePos' => 1355,
          ),
        ),
        'docComment' => '/**
 * @var float 3 * BETA
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 43,
      ),
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 55,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'aliasName' => NULL,
      ),
      'activate' => 
      array (
        'name' => 'activate',
        'parameters' => 
        array (
          'input' => 
          array (
            'name' => 'input',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'NDArray',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 71,
            'endLine' => 71,
            'startColumn' => 30,
            'endColumn' => 43,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'NDArray',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Apply the GeLU activation function to the input.
 *
 * f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 *
 * @param NDArray $input The input values
 * @return NDArray The activated values
 */',
        'startLine' => 71,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'aliasName' => NULL,
      ),
      'differentiate' => 
      array (
        'name' => 'differentiate',
        'parameters' => 
        array (
          'input' => 
          array (
            'name' => 'input',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'NDArray',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 99,
            'endLine' => 99,
            'startColumn' => 35,
            'endColumn' => 48,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'NDArray',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the derivative of the activation function.
 *
 * The derivative of GeLU is:
 * f\'(x) = 0.5 * (1 + tanh(α * (x + β * x^3))) +
 *         0.5 * x * sech^2(α * (x + β * x^3)) * α * (1 + 3β * x^2)
 *
 * Where:
 * - α = sqrt(2/π) ≈ 0.7978845608
 * - β = 0.044715
 * - sech^2(z) = (1/cosh(z))^2
 *
 * @param NDArray $input Input matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 99,
        'endLine' => 140,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the string representation of the activation function.
 *
 * @return string String representation
 */',
        'startLine' => 147,
        'endLine' => 150,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\GELU',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));