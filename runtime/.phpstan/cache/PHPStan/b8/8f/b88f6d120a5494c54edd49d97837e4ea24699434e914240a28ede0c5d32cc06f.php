<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/HardSigmoid.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-3bd232e7c18a00092901be35a59d3e3f546ecf3b4fd1972728e9677a1f3fedd4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/HardSigmoid.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
    'shortName' => 'HardSigmoid',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * HardSigmoid
 *
 * A piecewise linear approximation of the sigmoid function that is computationally
 * more efficient. The Hard Sigmoid function has an output value between 0 and 1,
 * making it useful for binary classification problems.
 *
 * f(x) = max(0, min(1, 0.2 * x + 0.5))
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
    'endLine' => 111,
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
      'SLOPE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'name' => 'SLOPE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.2',
          'attributes' => 
          array (
            'startLine' => 34,
            'endLine' => 34,
            'startTokenPos' => 65,
            'startFilePos' => 872,
            'endTokenPos' => 65,
            'endFilePos' => 874,
          ),
        ),
        'docComment' => '/**
 * The slope of the linear region.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 32,
      ),
      'INTERCEPT' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'name' => 'INTERCEPT',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0.5',
          'attributes' => 
          array (
            'startLine' => 41,
            'endLine' => 41,
            'startTokenPos' => 78,
            'startFilePos' => 996,
            'endTokenPos' => 78,
            'endFilePos' => 998,
          ),
        ),
        'docComment' => '/**
 * The y-intercept of the linear region.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 41,
        'endLine' => 41,
        'startColumn' => 5,
        'endColumn' => 36,
      ),
      'LOWER_BOUND' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'name' => 'LOWER_BOUND',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '-2.5',
          'attributes' => 
          array (
            'startLine' => 48,
            'endLine' => 48,
            'startTokenPos' => 91,
            'startFilePos' => 1122,
            'endTokenPos' => 92,
            'endFilePos' => 1125,
          ),
        ),
        'docComment' => '/**
 * The lower bound of the linear region.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 48,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 39,
      ),
      'UPPER_BOUND' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'name' => 'UPPER_BOUND',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2.5',
          'attributes' => 
          array (
            'startLine' => 55,
            'endLine' => 55,
            'startTokenPos' => 105,
            'startFilePos' => 1249,
            'endTokenPos' => 105,
            'endFilePos' => 1251,
          ),
        ),
        'docComment' => '/**
 * The upper bound of the linear region.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 55,
        'endLine' => 55,
        'startColumn' => 5,
        'endColumn' => 38,
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
        'startLine' => 57,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
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
            'startLine' => 73,
            'endLine' => 73,
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
 * Apply the HardSigmoid activation function to the input.
 *
 * f(x) = max(0, min(1, 0.2 * x + 0.5))
 *
 * @param NDArray $input The input values
 * @return NDArray The activated values
 */',
        'startLine' => 73,
        'endLine' => 81,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
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
            'startLine' => 92,
            'endLine' => 92,
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
 * f\'(x) = 0.2 if -2.5 <= x <= 2.5
 * f\'(x) = 0   otherwise
 *
 * @param NDArray $input Input matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 92,
        'endLine' => 100,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
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
        'startLine' => 107,
        'endLine' => 110,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\HardSigmoid',
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