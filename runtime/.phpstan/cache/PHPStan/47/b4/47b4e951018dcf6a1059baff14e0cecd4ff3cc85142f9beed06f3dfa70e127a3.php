<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/HardSiLU.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\HardSiLU
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-be3d8e5cf019853c69e2ec7794cb21d919988661b36e99ce4819f30df9763e3a',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/HardSiLU.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
    'shortName' => 'HardSiLU',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * HardSiLU
 *
 * Hard Sigmoid Linear Units (Hard SiLU) are a computationally efficient variant of the SiLU activation function.
 *
 * References:
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
 * Reinforcement Learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 27,
    'endLine' => 90,
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
    ),
    'immediateProperties' => 
    array (
      'hardSigmoid' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'name' => 'hardSigmoid',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The Hard Sigmoid activation function.
 *
 * @var HardSigmoid
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 39,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
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
        'docComment' => '/**
 * Class constructor.
 */',
        'startLine' => 39,
        'endLine' => 47,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
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
            'startLine' => 57,
            'endLine' => 57,
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
 * Apply the HardSiLU activation function to the input.
 *
 * f(x) = x * HardSigmoid(x)
 *
 * @param NDArray $input The input values
 * @return NDArray The activated values
 */',
        'startLine' => 57,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
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
            'startLine' => 72,
            'endLine' => 72,
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
 * f\'(x) = HardSigmoid(x) + x * HardSigmoid\'(x)
 *
 * @param NDArray $input Input matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 72,
        'endLine' => 79,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
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
        'startLine' => 86,
        'endLine' => 89,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSiLU',
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