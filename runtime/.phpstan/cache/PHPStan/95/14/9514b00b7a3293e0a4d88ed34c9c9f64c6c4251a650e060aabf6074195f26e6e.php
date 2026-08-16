<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Extractors/Deduplicator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Extractors\Deduplicator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-4c4eba485be58f90eae4b4be9cdc3b8c99897d087fc50604d2e890a63b4ec712',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Extractors/Deduplicator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Extractors',
    'name' => 'Rubix\\ML\\Extractors\\Deduplicator',
    'shortName' => 'Deduplicator',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Deduplicator
 *
 * Removes duplicate records from a dataset while the records are in flight. Deduplicator uses a memory-efficient
 * Bloom filter to probabilistically identify records that have already been seen before.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 20,
    'endLine' => 88,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Extractors\\Extractor',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'iterator' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'name' => 'iterator',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'iterable',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The base iterator.
 *
 * @var iterable<mixed[]>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 27,
        'endLine' => 27,
        'startColumn' => 5,
        'endColumn' => 33,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'filter' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'name' => 'filter',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'OkBloomer\\BloomFilter',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The Bloom filter.
 *
 * @var BloomFilter
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 34,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'dropped' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'name' => 'dropped',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 41,
            'endLine' => 41,
            'startTokenPos' => 64,
            'startFilePos' => 824,
            'endTokenPos' => 64,
            'endFilePos' => 824,
          ),
        ),
        'docComment' => '/**
 * The number of records that have been dropped so far.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 41,
        'endLine' => 41,
        'startColumn' => 5,
        'endColumn' => 31,
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
            'startLine' => 50,
            'endLine' => 50,
            'startColumn' => 9,
            'endColumn' => 26,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'maxFalsePositiveRate' => 
          array (
            'name' => 'maxFalsePositiveRate',
            'default' => 
            array (
              'code' => '0.001',
              'attributes' => 
              array (
                'startLine' => 51,
                'endLine' => 51,
                'startTokenPos' => 87,
                'startFilePos' => 1090,
                'endTokenPos' => 87,
                'endFilePos' => 1094,
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
            'startLine' => 51,
            'endLine' => 51,
            'startColumn' => 9,
            'endColumn' => 43,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'numHashes' => 
          array (
            'name' => 'numHashes',
            'default' => 
            array (
              'code' => '4',
              'attributes' => 
              array (
                'startLine' => 52,
                'endLine' => 52,
                'startTokenPos' => 97,
                'startFilePos' => 1123,
                'endTokenPos' => 97,
                'endFilePos' => 1123,
              ),
            ),
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
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 52,
            'endLine' => 52,
            'startColumn' => 9,
            'endColumn' => 27,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'layerSize' => 
          array (
            'name' => 'layerSize',
            'default' => 
            array (
              'code' => '32000000',
              'attributes' => 
              array (
                'startLine' => 53,
                'endLine' => 53,
                'startTokenPos' => 106,
                'startFilePos' => 1151,
                'endTokenPos' => 106,
                'endFilePos' => 1158,
              ),
            ),
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
            'startLine' => 53,
            'endLine' => 53,
            'startColumn' => 9,
            'endColumn' => 33,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param iterable<mixed[]> $iterator
 * @param float $maxFalsePositiveRate
 * @param int|null $numHashes
 * @param int $layerSize
 */',
        'startLine' => 49,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Extractors',
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'currentClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'aliasName' => NULL,
      ),
      'dropped' => 
      array (
        'name' => 'dropped',
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
 * Return the number of records that have been dropped so far.
 *
 * @return int
 */',
        'startLine' => 64,
        'endLine' => 67,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Extractors',
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'currentClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'aliasName' => NULL,
      ),
      'getIterator' => 
      array (
        'name' => 'getIterator',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Traversable',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an iterator for the records in the data table.
 *
 * @return \\Generator<mixed[]>
 */',
        'startLine' => 74,
        'endLine' => 87,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Extractors',
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
        'currentClassName' => 'Rubix\\ML\\Extractors\\Deduplicator',
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