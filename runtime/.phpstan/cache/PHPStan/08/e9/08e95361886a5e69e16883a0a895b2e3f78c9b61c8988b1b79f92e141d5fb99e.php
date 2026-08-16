<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Comparable.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Tensor\Comparable
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-18f956ee63d56c8931ebf86cd33d52bc1a4f979d938731309a1ac3ea5ce7ffdd-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Tensor\\Comparable',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Comparable.php',
      ),
    ),
    'namespace' => 'Tensor',
    'name' => 'Tensor\\Comparable',
    'shortName' => 'Comparable',
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
    'endLine' => 54,
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
      'equal' => 
      array (
        'name' => 'equal',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 13,
            'endLine' => 13,
            'startColumn' => 27,
            'endColumn' => 28,
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
 * A universal function to compute the equality comparison of a tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 13,
        'endLine' => 13,
        'startColumn' => 5,
        'endColumn' => 30,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
        'aliasName' => NULL,
      ),
      'notEqual' => 
      array (
        'name' => 'notEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 21,
            'endLine' => 21,
            'startColumn' => 30,
            'endColumn' => 31,
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
 * A universal function to compute the not equal comparison of this tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 21,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 33,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
        'aliasName' => NULL,
      ),
      'greater' => 
      array (
        'name' => 'greater',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 29,
            'endLine' => 29,
            'startColumn' => 29,
            'endColumn' => 30,
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
 * A universal function to compute the greater than comparison of a tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 29,
        'endLine' => 29,
        'startColumn' => 5,
        'endColumn' => 32,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
        'aliasName' => NULL,
      ),
      'greaterEqual' => 
      array (
        'name' => 'greaterEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 37,
            'endLine' => 37,
            'startColumn' => 34,
            'endColumn' => 35,
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
 * A universal function to compute the greater than or equal to comparison of a tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 37,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
        'aliasName' => NULL,
      ),
      'less' => 
      array (
        'name' => 'less',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 45,
            'endLine' => 45,
            'startColumn' => 26,
            'endColumn' => 27,
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
 * A universal function to compute the less than comparison of a tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 29,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
        'aliasName' => NULL,
      ),
      'lessEqual' => 
      array (
        'name' => 'lessEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 53,
            'endLine' => 53,
            'startColumn' => 31,
            'endColumn' => 32,
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
 * A universal function to compute the less than or equal to comparison of a tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 34,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Comparable',
        'implementingClassName' => 'Tensor\\Comparable',
        'currentClassName' => 'Tensor\\Comparable',
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