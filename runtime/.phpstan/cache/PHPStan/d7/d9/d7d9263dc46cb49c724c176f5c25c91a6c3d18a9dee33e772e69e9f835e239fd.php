<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/LeakyReLU.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-7279a3dc0190d488a073938a05772d2a59d0da2de9993a5f7b7882e17f2bd628',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/LeakyReLU.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
    'shortName' => 'LeakyReLU',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Leaky ReLU
 *
 * Leaky Rectified Linear Units are functions that output x when x > 0 or a
 * small leakage value when x < 0. The amount of leakage is controlled by the
 * user-specified parameter.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network
 * Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 30,
    'endLine' => 112,
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
      'leakage' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'name' => 'leakage',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The amount of leakage as a ratio of the input value to allow to pass through when inactivated.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 29,
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
          'leakage' => 
          array (
            'name' => 'leakage',
            'default' => 
            array (
              'code' => '0.1',
              'attributes' => 
              array (
                'startLine' => 45,
                'endLine' => 45,
                'startTokenPos' => 83,
                'startFilePos' => 1282,
                'endTokenPos' => 83,
                'endFilePos' => 1284,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 45,
            'endLine' => 45,
            'startColumn' => 33,
            'endColumn' => 52,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Class constructor.
 *
 * @param float $leakage The amount of leakage as a ratio of the input value to allow to pass through when inactivated.
 * @throws InvalidLeakageException
 */',
        'startLine' => 45,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
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
            'startLine' => 70,
            'endLine' => 70,
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
 * Apply the Leaky ReLU activation function to the input.
 *
 * f(x) = x           if x > 0
 * f(x) = leakage * x if x ≤ 0
 *
 * @param NDArray $input The input values
 * @return NDArray The activated values
 */',
        'startLine' => 70,
        'endLine' => 80,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
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
            'startLine' => 91,
            'endLine' => 91,
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
 * f\'(x) = 1         if x > 0
 * f\'(x) = leakage   if x ≤ 0
 *
 * @param NDArray $input Input matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 91,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
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
        'startLine' => 108,
        'endLine' => 111,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\LeakyReLU',
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