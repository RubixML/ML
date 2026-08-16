<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/functions.php-PHPStan\BetterReflection\Reflection\ReflectionFunction-Rubix\ML\iterator_filter
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1a4a61157c47de52bea1079cacf2ded805a853d420f5790f53125682c8ffedcf',
   'data' => 
  array (
    'name' => 'iterator_filter',
    'parameters' => 
    array (
      'iterator' => 
      array (
        'name' => 'iterator',
        'default' => NULL,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'iterable',
            'isIdentifier' => true,
          ),
        ),
        'isVariadic' => false,
        'byRef' => false,
        'isPromoted' => false,
        'attributes' => 
        array (
        ),
        'startLine' => 203,
        'endLine' => 203,
        'startColumn' => 30,
        'endColumn' => 47,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'callback' => 
      array (
        'name' => 'callback',
        'default' => NULL,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'callable',
            'isIdentifier' => true,
          ),
        ),
        'isVariadic' => false,
        'byRef' => false,
        'isPromoted' => false,
        'attributes' => 
        array (
        ),
        'startLine' => 203,
        'endLine' => 203,
        'startColumn' => 50,
        'endColumn' => 67,
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
        'name' => 'Generator',
        'isIdentifier' => false,
      ),
    ),
    'attributes' => 
    array (
    ),
    'docComment' => '/**
 * Filter the elements of an iterator using a callback.
 *
 * @internal
 *
 * @param iterable<mixed> $iterator
 * @param callable $callback
 * @return Generator<mixed>
 */',
    'startLine' => 203,
    'endLine' => 210,
    'startColumn' => 5,
    'endColumn' => 5,
    'couldThrow' => false,
    'isClosure' => false,
    'isGenerator' => true,
    'isVariadic' => false,
    'isStatic' => false,
    'namespace' => 'Rubix\\ML',
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\iterator_filter',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/functions.php',
      ),
    ),
  ),
));