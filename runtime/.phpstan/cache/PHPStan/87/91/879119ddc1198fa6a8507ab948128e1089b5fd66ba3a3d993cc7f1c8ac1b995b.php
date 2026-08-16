<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Serializers/GzipNative.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Serializers\GzipNative
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-92e7576808b6da30f21d4f85e01a9dc7e4fe7b17ccedf19e9f03997ee5b5475e',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Serializers\\GzipNative',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Serializers/GzipNative.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Serializers',
    'name' => 'Rubix\\ML\\Serializers\\GzipNative',
    'shortName' => 'GzipNative',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Gzip Native
 *
 * Gzip Native wraps the native PHP serialization format in an outer compression layer based on the
 * DEFLATE algorithm with a header and CRC32 checksum.
 *
 * References:
 * [1] P. Deutsch. (1996). RFC 1951 - DEFLATE Compressed Data Format Specification version.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 23,
    'endLine' => 114,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Serializers\\Serializer',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'level' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'name' => 'level',
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
        'default' => NULL,
        'docComment' => '/**
 * The compression level between 0 and 9, 0 meaning no compression.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 30,
        'endLine' => 30,
        'startColumn' => 5,
        'endColumn' => 25,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'base' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'name' => 'base',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Serializers\\Native',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The base serializer.
 *
 * @var Native
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 27,
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
          'level' => 
          array (
            'name' => 'level',
            'default' => 
            array (
              'code' => '6',
              'attributes' => 
              array (
                'startLine' => 43,
                'endLine' => 43,
                'startTokenPos' => 71,
                'startFilePos' => 969,
                'endTokenPos' => 71,
                'endFilePos' => 969,
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
            'startLine' => 43,
            'endLine' => 43,
            'startColumn' => 33,
            'endColumn' => 46,
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
 * @param int $level
 * @throws InvalidArgumentException
 */',
        'startLine' => 43,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'currentClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'aliasName' => NULL,
      ),
      'level' => 
      array (
        'name' => 'level',
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
 * Return the level of compression between 0 and 9.
 *
 * @internal
 *
 * @return int
 */',
        'startLine' => 61,
        'endLine' => 64,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'currentClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'aliasName' => NULL,
      ),
      'serialize' => 
      array (
        'name' => 'serialize',
        'parameters' => 
        array (
          'persistable' => 
          array (
            'name' => 'persistable',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Persistable',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 72,
            'endLine' => 72,
            'startColumn' => 31,
            'endColumn' => 54,
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
            'name' => 'Rubix\\ML\\Encoding',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Serialize a persistable object and return the data.
 *
 * @param Persistable $persistable
 * @return Encoding
 */',
        'startLine' => 72,
        'endLine' => 83,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'currentClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'aliasName' => NULL,
      ),
      'deserialize' => 
      array (
        'name' => 'deserialize',
        'parameters' => 
        array (
          'encoding' => 
          array (
            'name' => 'encoding',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Encoding',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 92,
            'endLine' => 92,
            'startColumn' => 33,
            'endColumn' => 50,
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
            'name' => 'Rubix\\ML\\Persistable',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Deserialize a persistable object and return it.
 *
 * @param Encoding $encoding
 * @throws RuntimeException
 * @return Persistable
 */',
        'startLine' => 92,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'currentClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
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
        'startLine' => 110,
        'endLine' => 113,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
        'currentClassName' => 'Rubix\\ML\\Serializers\\GzipNative',
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