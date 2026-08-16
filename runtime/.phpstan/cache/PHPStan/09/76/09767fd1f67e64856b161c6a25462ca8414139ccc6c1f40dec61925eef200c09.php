<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/OBufferDerivative.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\ActivationFunctions\OBufferDerivative
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-23f958bdde1b9ccabbb68b3d71a754dcccd05ef6c588174029d448068ee54cda',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\OBufferDerivative',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/ActivationFunctions/OBufferDerivative.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\OBufferDerivative',
    'shortName' => 'OBufferDerivative',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Derivative based on output buffer
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
    'startLine' => 16,
    'endLine' => 25,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'differentiate' => 
      array (
        'name' => 'differentiate',
        'parameters' => 
        array (
          'output' => 
          array (
            'name' => 'output',
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
            'startLine' => 24,
            'endLine' => 24,
            'startColumn' => 35,
            'endColumn' => 49,
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
 * Calculate the derivative of the activation.
 *
 * @param NDArray $output Output matrix
 * @return NDArray Derivative matrix
 */',
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 61,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\OBufferDerivative',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\OBufferDerivative',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\ActivationFunctions\\OBufferDerivative',
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