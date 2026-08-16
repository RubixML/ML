<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Serializers/RBX.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Serializers\RBX
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-a5b059a47d32f8092de50fa4aa437d1a1f8f3ab135a7ce1e2bd3ab7cc55546aa',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Serializers\\RBX',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Serializers/RBX.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Serializers',
    'name' => 'Rubix\\ML\\Serializers\\RBX',
    'shortName' => 'RBX',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * RBX
 *
 * Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP\'s native
 * serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all
 * in one robust format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 31,
    'endLine' => 199,
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
      'IDENTIFIER_STRING' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'name' => 'IDENTIFIER_STRING',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '"\\xa1RBX\\r\\n\\x1a\\n"',
          'attributes' => 
          array (
            'startLine' => 38,
            'endLine' => 38,
            'startTokenPos' => 107,
            'startFilePos' => 950,
            'endTokenPos' => 107,
            'endFilePos' => 968,
          ),
        ),
        'docComment' => '/**
 * The identifier or "magic number" of the format.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 60,
      ),
      'VERSION' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'name' => 'VERSION',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 45,
            'endLine' => 45,
            'startTokenPos' => 120,
            'startFilePos' => 1083,
            'endTokenPos' => 120,
            'endFilePos' => 1083,
          ),
        ),
        'docComment' => '/**
 * The current version of the format.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 32,
      ),
      'CHECKSUM_HASH_TYPE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'name' => 'CHECKSUM_HASH_TYPE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'crc32b\'',
          'attributes' => 
          array (
            'startLine' => 52,
            'endLine' => 52,
            'startTokenPos' => 133,
            'startFilePos' => 1226,
            'endTokenPos' => 133,
            'endFilePos' => 1233,
          ),
        ),
        'docComment' => '/**
 * The hashing function used to generate checksums.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 52,
        'endLine' => 52,
        'startColumn' => 5,
        'endColumn' => 50,
      ),
      'EOL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'name' => 'EOL',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '"\\n"',
          'attributes' => 
          array (
            'startLine' => 59,
            'endLine' => 59,
            'startTokenPos' => 146,
            'startFilePos' => 1339,
            'endTokenPos' => 146,
            'endFilePos' => 1342,
          ),
        ),
        'docComment' => '/**
 * The end of line character.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 59,
        'endLine' => 59,
        'startColumn' => 5,
        'endColumn' => 31,
      ),
    ),
    'immediateProperties' => 
    array (
      'base' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'name' => 'base',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Serializers\\GzipNative',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The base Gzip Native serializer.
 *
 * @var GzipNative
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 66,
        'endLine' => 66,
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
          'level' => 
          array (
            'name' => 'level',
            'default' => 
            array (
              'code' => '6',
              'attributes' => 
              array (
                'startLine' => 71,
                'endLine' => 71,
                'startTokenPos' => 172,
                'startFilePos' => 1551,
                'endTokenPos' => 172,
                'endFilePos' => 1551,
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
            'startLine' => 71,
            'endLine' => 71,
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
 */',
        'startLine' => 71,
        'endLine' => 74,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'currentClassName' => 'Rubix\\ML\\Serializers\\RBX',
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
            'startLine' => 84,
            'endLine' => 84,
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
 * @internal
 *
 * @param Persistable $persistable
 * @return Encoding
 */',
        'startLine' => 84,
        'endLine' => 118,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'currentClassName' => 'Rubix\\ML\\Serializers\\RBX',
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
            'startLine' => 129,
            'endLine' => 129,
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
 * @internal
 *
 * @param Encoding $encoding
 * @throws RuntimeException
 * @return Persistable
 */',
        'startLine' => 129,
        'endLine' => 186,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'currentClassName' => 'Rubix\\ML\\Serializers\\RBX',
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
        'startLine' => 195,
        'endLine' => 198,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\RBX',
        'currentClassName' => 'Rubix\\ML\\Serializers\\RBX',
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