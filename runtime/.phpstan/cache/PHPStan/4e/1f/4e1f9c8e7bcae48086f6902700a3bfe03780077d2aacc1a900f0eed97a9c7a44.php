<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/CostFunctions/MeanAbsoluteError.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\CostFunctions\MeanAbsoluteError
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-3e049db55304cfd0bd8ce7e378223653e13f75bf3b5f4a5773f37fc53fc4923e',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/CostFunctions/MeanAbsoluteError.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\CostFunctions',
    'name' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
    'shortName' => 'MeanAbsoluteError',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Mean Absolute Error
 *
 * Mean Absolute Error (MAE) measures the average magnitude of errors between
 * predicted and actual values without considering their direction. It is a
 * linear score which means all individual differences are weighted equally.
 * MAE is more robust to outliers compared to Mean Squared Error.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 26,
    'endLine' => 84,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\CostFunctions\\RegressionLoss',
    ),
    'traitClassNames' => 
    array (
      0 => 'Rubix\\ML\\Traits\\AssertsShapes',
    ),
    'immediateConstants' => 
    array (
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
        'startLine' => 30,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\CostFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'aliasName' => NULL,
      ),
      'compute' => 
      array (
        'name' => 'compute',
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
            'startLine' => 47,
            'endLine' => 47,
            'startColumn' => 29,
            'endColumn' => 43,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'target' => 
          array (
            'name' => 'target',
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
            'startLine' => 47,
            'endLine' => 47,
            'startColumn' => 46,
            'endColumn' => 60,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the loss score.
 *
 * L(y, ŷ) = Σ|y - ŷ| / n
 *
 * @param NDArray $output The output of the network
 * @param NDArray $target The target values
 * @return float
 */',
        'startLine' => 47,
        'endLine' => 55,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\CostFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'aliasName' => NULL,
      ),
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
            'startLine' => 66,
            'endLine' => 66,
            'startColumn' => 35,
            'endColumn' => 49,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'target' => 
          array (
            'name' => 'target',
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
            'startLine' => 66,
            'endLine' => 66,
            'startColumn' => 52,
            'endColumn' => 66,
            'parameterIndex' => 1,
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
 * Calculate the gradient of the cost function with respect to the output.
 *
 * ∂L/∂ŷ = sign(ŷ - y)
 *
 * @param NDArray $output The output of the network
 * @param NDArray $target The target values
 * @return NDArray
 */',
        'startLine' => 66,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\CostFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
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
 * Return the string representation of the object.
 *
 * @return string
 */',
        'startLine' => 80,
        'endLine' => 83,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\CostFunctions',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\CostFunctions\\MeanAbsoluteError',
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