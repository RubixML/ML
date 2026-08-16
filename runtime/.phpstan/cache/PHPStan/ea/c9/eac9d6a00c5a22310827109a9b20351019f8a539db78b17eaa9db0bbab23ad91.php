<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Traits/AssertsShapes.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Traits\AssertsShapes
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-9974eb05eaeedb670507b479abd6f4618fa0865a05fe485f3256755759b63246',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Traits\\AssertsShapes',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Traits/AssertsShapes.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Traits',
    'name' => 'Rubix\\ML\\Traits\\AssertsShapes',
    'shortName' => 'AssertsShapes',
    'isInterface' => false,
    'isTrait' => true,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Validates Shapes
 *
 * A trait that provides shape validation for cost functions to ensure
 * output and target arrays have matching dimensions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 20,
    'endLine' => 35,
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
      'assertSameShape' => 
      array (
        'name' => 'assertSameShape',
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
            'startLine' => 29,
            'endLine' => 29,
            'startColumn' => 40,
            'endColumn' => 54,
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
            'startLine' => 29,
            'endLine' => 29,
            'startColumn' => 57,
            'endColumn' => 71,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Assert that the output and target NDArrays have identical shapes.
 *
 * @param NDArray $output The output array to check.
 * @param NDArray $target The target array to compare against.
 * @throws InvalidArgumentException If the shapes do not match.
 */',
        'startLine' => 29,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 2,
        'namespace' => 'Rubix\\ML\\Traits',
        'declaringClassName' => 'Rubix\\ML\\Traits\\AssertsShapes',
        'implementingClassName' => 'Rubix\\ML\\Traits\\AssertsShapes',
        'currentClassName' => 'Rubix\\ML\\Traits\\AssertsShapes',
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