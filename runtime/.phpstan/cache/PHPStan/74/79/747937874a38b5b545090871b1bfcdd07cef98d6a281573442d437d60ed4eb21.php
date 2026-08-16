<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Statistical.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Tensor\Statistical
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-7d5988214fcf945193d3f6d8bd26076ed828d0950a3263cb9b0dd2a2fe748ddd-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Tensor\\Statistical',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Statistical.php',
      ),
    ),
    'namespace' => 'Tensor',
    'name' => 'Tensor\\Statistical',
    'shortName' => 'Statistical',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => NULL,
    'attributes' => 
    array (
    ),
    'startLine' => 5,
    'endLine' => 36,
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
      'mean' => 
      array (
        'name' => 'mean',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the mean of the tensor.
 *
 * @return mixed
 */',
        'startLine' => 12,
        'endLine' => 12,
        'startColumn' => 5,
        'endColumn' => 27,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Statistical',
        'implementingClassName' => 'Tensor\\Statistical',
        'currentClassName' => 'Tensor\\Statistical',
        'aliasName' => NULL,
      ),
      'variance' => 
      array (
        'name' => 'variance',
        'parameters' => 
        array (
          'mean' => 
          array (
            'name' => 'mean',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 20,
                'endLine' => 20,
                'startTokenPos' => 36,
                'startFilePos' => 311,
                'endTokenPos' => 36,
                'endFilePos' => 314,
              ),
            ),
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 20,
            'endLine' => 20,
            'startColumn' => 30,
            'endColumn' => 41,
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
 * Compute the variance of the tensor.
 *
 * @param mixed $mean
 * @return mixed
 */',
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 43,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Statistical',
        'implementingClassName' => 'Tensor\\Statistical',
        'currentClassName' => 'Tensor\\Statistical',
        'aliasName' => NULL,
      ),
      'median' => 
      array (
        'name' => 'median',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the median of the tensor.
 *
 * @return mixed
 */',
        'startLine' => 27,
        'endLine' => 27,
        'startColumn' => 5,
        'endColumn' => 29,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Statistical',
        'implementingClassName' => 'Tensor\\Statistical',
        'currentClassName' => 'Tensor\\Statistical',
        'aliasName' => NULL,
      ),
      'quantile' => 
      array (
        'name' => 'quantile',
        'parameters' => 
        array (
          'q' => 
          array (
            'name' => 'q',
            'default' => NULL,
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
            'startLine' => 35,
            'endLine' => 35,
            'startColumn' => 30,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the q\'th quantile of the tensor.
 *
 * @param float $q
 * @return mixed
 */',
        'startLine' => 35,
        'endLine' => 35,
        'startColumn' => 5,
        'endColumn' => 39,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Statistical',
        'implementingClassName' => 'Tensor\\Statistical',
        'currentClassName' => 'Tensor\\Statistical',
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