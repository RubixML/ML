<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Helpers/JSON.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Helpers\JSON
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-c4c352f443d3cfc09ef714b1fbae31129587345f327b213f372baaab7ecb5f07',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Helpers\\JSON',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Helpers/JSON.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Helpers',
    'name' => 'Rubix\\ML\\Helpers\\JSON',
    'shortName' => 'JSON',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * JSON
 *
 * A helper class providing functions relating to JSON (Javascript Object Notation).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 22,
    'endLine' => 77,
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
      'DEFAULT_DEPTH' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'name' => 'DEFAULT_DEPTH',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '512',
          'attributes' => 
          array (
            'startLine' => 29,
            'endLine' => 29,
            'startTokenPos' => 58,
            'startFilePos' => 510,
            'endTokenPos' => 58,
            'endFilePos' => 512,
          ),
        ),
        'docComment' => '/**
 * The default maximum stack depth.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 29,
        'endLine' => 29,
        'startColumn' => 5,
        'endColumn' => 40,
      ),
      'DEFAULT_OPTIONS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'name' => 'DEFAULT_OPTIONS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 34,
            'endLine' => 34,
            'startTokenPos' => 71,
            'startFilePos' => 586,
            'endTokenPos' => 71,
            'endFilePos' => 586,
          ),
        ),
        'docComment' => '/**
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 40,
      ),
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'encode' => 
      array (
        'name' => 'encode',
        'parameters' => 
        array (
          'value' => 
          array (
            'name' => 'value',
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
            'startColumn' => 35,
            'endColumn' => 40,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'options' => 
          array (
            'name' => 'options',
            'default' => 
            array (
              'code' => 'self::DEFAULT_OPTIONS',
              'attributes' => 
              array (
                'startLine' => 45,
                'endLine' => 45,
                'startTokenPos' => 93,
                'startFilePos' => 835,
                'endTokenPos' => 95,
                'endFilePos' => 855,
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
            'startLine' => 45,
            'endLine' => 45,
            'startColumn' => 43,
            'endColumn' => 78,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'depth' => 
          array (
            'name' => 'depth',
            'default' => 
            array (
              'code' => 'self::DEFAULT_DEPTH',
              'attributes' => 
              array (
                'startLine' => 45,
                'endLine' => 45,
                'startTokenPos' => 104,
                'startFilePos' => 871,
                'endTokenPos' => 106,
                'endFilePos' => 889,
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
            'startLine' => 45,
            'endLine' => 45,
            'startColumn' => 81,
            'endColumn' => 112,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
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
 * Serialize to JSON.
 *
 * @param mixed $value
 * @param int $options
 * @param positive-int $depth
 * @throws JSONException
 * @return string
 */',
        'startLine' => 45,
        'endLine' => 54,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Helpers',
        'declaringClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'currentClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'aliasName' => NULL,
      ),
      'decode' => 
      array (
        'name' => 'decode',
        'parameters' => 
        array (
          'data' => 
          array (
            'name' => 'data',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'string',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 65,
            'endLine' => 65,
            'startColumn' => 35,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'options' => 
          array (
            'name' => 'options',
            'default' => 
            array (
              'code' => 'self::DEFAULT_OPTIONS',
              'attributes' => 
              array (
                'startLine' => 65,
                'endLine' => 65,
                'startTokenPos' => 185,
                'startFilePos' => 1346,
                'endTokenPos' => 187,
                'endFilePos' => 1366,
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
            'startLine' => 65,
            'endLine' => 65,
            'startColumn' => 49,
            'endColumn' => 84,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'depth' => 
          array (
            'name' => 'depth',
            'default' => 
            array (
              'code' => 'self::DEFAULT_DEPTH',
              'attributes' => 
              array (
                'startLine' => 65,
                'endLine' => 65,
                'startTokenPos' => 196,
                'startFilePos' => 1382,
                'endTokenPos' => 198,
                'endFilePos' => 1400,
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
            'startLine' => 65,
            'endLine' => 65,
            'startColumn' => 87,
            'endColumn' => 118,
            'parameterIndex' => 2,
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
 * Deserialize from JSON.
 *
 * @param string $data
 * @param int $options
 * @param positive-int $depth
 * @throws JSONException
 * @return mixed[]
 */',
        'startLine' => 65,
        'endLine' => 76,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Helpers',
        'declaringClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'implementingClassName' => 'Rubix\\ML\\Helpers\\JSON',
        'currentClassName' => 'Rubix\\ML\\Helpers\\JSON',
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