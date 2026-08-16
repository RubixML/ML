<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionFunction-iterator_to_array
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'name' => 'iterator_to_array',
    'parameters' => 
    array (
      'iterator' => 
      array (
        'name' => 'iterator',
        'default' => NULL,
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
                  'name' => 'Traversable',
                  'isIdentifier' => false,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'array',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'isVariadic' => false,
        'byRef' => false,
        'isPromoted' => false,
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '[\'8.2\' => \'Traversable|array\']',
                'attributes' => 
                array (
                  'startLine' => 22,
                  'endLine' => 22,
                  'startTokenPos' => 16,
                  'startFilePos' => 762,
                  'endTokenPos' => 22,
                  'endFilePos' => 791,
                ),
              ),
              'default' => 
              array (
                'code' => '\'Traversable\'',
                'attributes' => 
                array (
                  'startLine' => 22,
                  'endLine' => 22,
                  'startTokenPos' => 28,
                  'startFilePos' => 803,
                  'endTokenPos' => 28,
                  'endFilePos' => 815,
                ),
              ),
            ),
          ),
        ),
        'startLine' => 22,
        'endLine' => 23,
        'startColumn' => 9,
        'endColumn' => 35,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'preserve_keys' => 
      array (
        'name' => 'preserve_keys',
        'default' => 
        array (
          'code' => '\\true',
          'attributes' => 
          array (
            'startLine' => 24,
            'endLine' => 24,
            'startTokenPos' => 45,
            'startFilePos' => 886,
            'endTokenPos' => 45,
            'endFilePos' => 889,
          ),
        ),
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
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
        'startColumn' => 9,
        'endColumn' => 34,
        'parameterIndex' => 1,
        'isOptional' => true,
      ),
    ),
    'returnsReference' => false,
    'returnType' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
      'data' => 
      array (
        'name' => 'array',
        'isIdentifier' => true,
      ),
    ),
    'attributes' => 
    array (
    ),
    'docComment' => '/**
 * Copy the iterator into an array
 * @link https://php.net/manual/en/function.iterator-to-array.php
 * @template TKey of int|string
 * @template TValue
 * @param Traversable<TKey, TValue>|array<TKey, TValue> $iterator <p>
 * The iterator being copied.
 * </p>
 * @param bool $preserve_keys [optional] <p>
 * Whether to use the iterator element keys as index.
 * </p>
 * @return (
 *     $preserve_keys is true ? array<TKey, TValue> : (
 *     $preserve_keys is false ? TValue[] :
 *     array<TKey, TValue>|TValue[])
 * ) An array containing the elements of the iterator.
 */',
    'startLine' => 21,
    'endLine' => 27,
    'startColumn' => 5,
    'endColumn' => 5,
    'couldThrow' => false,
    'isClosure' => false,
    'isGenerator' => false,
    'isVariadic' => false,
    'isStatic' => false,
    'namespace' => NULL,
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'iterator_to_array',
        'filename' => 'phpstorm-stubs:SPL/SPL_f.stub',
        'extensionName' => 'SPL',
        'aliasName' => NULL,
      ),
    ),
  ),
));