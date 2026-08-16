<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Layers/Dropout.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\NeuralNet\Layers\Dropout
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-ae115d4e684a92e2e9aab1711584a4750988dc6e31aaf1154cb14604f6e6373a',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/NeuralNet/Layers/Dropout.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
    'name' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
    'shortName' => 'Dropout',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Dropout
 *
 * Dropout is a regularization technique for reducing overfitting in neural
 * networks by preventing complex co-adaptations on training data. It works
 * by temporarily disabling neurons during each training pass. It also is a
 * very efficient way of performing model averaging with neural networks.
 *
 * References:
 * [1] N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural
 * Networks from Overfitting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 32,
    'endLine' => 208,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\NeuralNet\\Layers\\Hidden',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'ratio' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'name' => 'ratio',
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
 * The ratio of neurons that are dropped during each training pass.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 39,
        'endLine' => 39,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'scale' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'name' => 'scale',
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
 * The scaling coefficient.
 *
 * @var float
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 46,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'width' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'name' => 'width',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'int',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 53,
            'endLine' => 53,
            'startTokenPos' => 93,
            'startFilePos' => 1363,
            'endTokenPos' => 93,
            'endFilePos' => 1366,
          ),
        ),
        'docComment' => '/**
 * The width of the layer.
 *
 * @var positive-int|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 33,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'mask' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'name' => 'mask',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'NDArray',
                  'isIdentifier' => false,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 60,
            'endLine' => 60,
            'startTokenPos' => 107,
            'startFilePos' => 1483,
            'endTokenPos' => 107,
            'endFilePos' => 1486,
          ),
        ),
        'docComment' => '/**
 * The memoized dropout mask.
 *
 * @var NDArray|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 60,
        'endLine' => 60,
        'startColumn' => 5,
        'endColumn' => 36,
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
          'ratio' => 
          array (
            'name' => 'ratio',
            'default' => 
            array (
              'code' => '0.5',
              'attributes' => 
              array (
                'startLine' => 66,
                'endLine' => 66,
                'startTokenPos' => 124,
                'startFilePos' => 1620,
                'endTokenPos' => 124,
                'endFilePos' => 1622,
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
            'startLine' => 66,
            'endLine' => 66,
            'startColumn' => 33,
            'endColumn' => 50,
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
 * @param float $ratio
 * @throws InvalidArgumentException
 */',
        'startLine' => 66,
        'endLine' => 79,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'width' => 
      array (
        'name' => 'width',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the width of the layer.
 *
 * @internal
 *
 * @throws RuntimeException
 * @return positive-int
 */',
        'startLine' => 89,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'initialize' => 
      array (
        'name' => 'initialize',
        'parameters' => 
        array (
          'fanIn' => 
          array (
            'name' => 'fanIn',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 107,
            'endLine' => 107,
            'startColumn' => 32,
            'endColumn' => 41,
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
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Initialize the layer with the fan in from the previous layer and return
 * the fan out for this layer.
 *
 * @internal
 *
 * @param positive-int $fanIn
 * @return positive-int
 */',
        'startLine' => 107,
        'endLine' => 114,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'forward' => 
      array (
        'name' => 'forward',
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
            'startLine' => 124,
            'endLine' => 124,
            'startColumn' => 29,
            'endColumn' => 42,
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
 * Compute a forward pass through the layer.
 *
 * @internal
 *
 * @param NDArray $input
 * @return NDArray
 */',
        'startLine' => 124,
        'endLine' => 142,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'infer' => 
      array (
        'name' => 'infer',
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
            'startLine' => 152,
            'endLine' => 152,
            'startColumn' => 27,
            'endColumn' => 40,
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
 * Compute an inferential pass through the layer.
 *
 * @internal
 *
 * @param NDArray $input
 * @return NDArray
 */',
        'startLine' => 152,
        'endLine' => 155,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'back' => 
      array (
        'name' => 'back',
        'parameters' => 
        array (
          'prevGradient' => 
          array (
            'name' => 'prevGradient',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Deferred',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 167,
            'endLine' => 167,
            'startColumn' => 26,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'optimizer' => 
          array (
            'name' => 'optimizer',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\NeuralNet\\Optimizers\\Optimizer',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 167,
            'endLine' => 167,
            'startColumn' => 50,
            'endColumn' => 69,
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
            'name' => 'Rubix\\ML\\Deferred',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the gradients of the layer and update the parameters.
 *
 * @internal
 *
 * @param Deferred $prevGradient
 * @param Optimizer $optimizer
 * @throws RuntimeException
 * @return Deferred
 */',
        'startLine' => 167,
        'endLine' => 178,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'aliasName' => NULL,
      ),
      'gradient' => 
      array (
        'name' => 'gradient',
        'parameters' => 
        array (
          'prevGradient' => 
          array (
            'name' => 'prevGradient',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Deferred',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 189,
            'endLine' => 189,
            'startColumn' => 30,
            'endColumn' => 51,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'mask' => 
          array (
            'name' => 'mask',
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
            'startLine' => 189,
            'endLine' => 189,
            'startColumn' => 54,
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
 * Calculate the gradient for the previous layer.
 *
 * @internal
 *
 * @param Deferred $prevGradient
 * @param NDArray $mask
 * @return NDArray
 */',
        'startLine' => 189,
        'endLine' => 195,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
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
 * @internal
 *
 * @return string
 */',
        'startLine' => 204,
        'endLine' => 207,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\NeuralNet\\Layers',
        'declaringClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'implementingClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
        'currentClassName' => 'Rubix\\ML\\NeuralNet\\Layers\\Dropout',
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