<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/SELU.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\SELU
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-00cf2e25aeb11de83334ae0d0d28878d8bab810af5a1c302c4aa662c5bfbf175',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/SELU.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
    'shortName' => 'SELU',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * SELU
 *
 * Scaled Exponential Linear Unit is a self-normalizing activation function
 * based on the ELU activation function. Neuronal activations of SELU networks
 * automatically converge toward zero mean and unit variance, unlike explicitly
 * normalized networks such as those with [Batch Norm](#batch-norm).
 *
 * References:
 * [1] G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 29,
    'endLine' => 122,
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
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'name' => 'ALPHA',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1.6732632',
          'attributes' => 
          array (
            'startLine' => 36,
            'endLine' => 36,
            'startTokenPos' => 65,
            'startFilePos' => 996,
            'endTokenPos' => 65,
            'endFilePos' => 1004,
          ),
        ),
        'docComment' => '/**
 * The value at which leakage starts to saturate.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 36,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 35,
      ),
      'LAMBDA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'name' => 'LAMBDA',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1.0507009',
          'attributes' => 
          array (
            'startLine' => 43,
            'endLine' => 43,
            'startTokenPos' => 78,
            'startFilePos' => 1107,
            'endTokenPos' => 78,
            'endFilePos' => 1115,
          ),
        ),
        'docComment' => '/**
 * The scaling coefficient.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 43,
        'endLine' => 43,
        'startColumn' => 5,
        'endColumn' => 36,
      ),
      'BETA' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'name' => 'BETA',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => 'self::LAMBDA * self::ALPHA',
          'attributes' => 
          array (
            'startLine' => 50,
            'endLine' => 50,
            'startTokenPos' => 91,
            'startFilePos' => 1239,
            'endTokenPos' => 99,
            'endFilePos' => 1264,
          ),
        ),
        'docComment' => '/**
 * The scaling coefficient multiplied by alpha.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 50,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 54,
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
        'startLine' => 52,
        'endLine' => 58,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
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
            'startLine' => 69,
            'endLine' => 69,
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
 * Compute the activation.
 *
 * f(x) = λ * x                 if x > 0
 * f(x) = λ * α * (e^x - 1)     if x ≤ 0
 *
 * @param NDArray $input The input values
 * @return NDArray The activated values
 */',
        'startLine' => 69,
        'endLine' => 83,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
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
            'startLine' => 94,
            'endLine' => 94,
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
 * Calculate the derivative of the SELU activation function.
 *
 * f\'(x) = λ                if x > 0
 * f\'(x) = λ * α * e^x      if x ≤ 0
 *
 * @param NDArray $input Input matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 94,
        'endLine' => 111,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
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
        'startLine' => 118,
        'endLine' => 121,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\SELU',
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