<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionClass-serializable
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'Serializable',
        'filename' => 'phpstorm-stubs:Core/Core_c.stub',
        'extensionName' => 'Core',
        'aliasName' => NULL,
      ),
    ),
    'namespace' => NULL,
    'name' => 'Serializable',
    'shortName' => 'Serializable',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Interface for customized serializing.<br>
 * As of PHP 8.1.0, a class which implements Serializable without also implementing `__serialize()` and `__unserialize()`
 * will generate a deprecation warning.
 * @link https://php.net/manual/en/class.serializable.php
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 10,
    'endLine' => 29,
    'startColumn' => 5,
    'endColumn' => 5,
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
      'serialize' => 
      array (
        'name' => 'serialize',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * String representation of object.
 * @link https://php.net/manual/en/serializable.serialize.php
 * @return string|null The string representation of the object or null
 * @throws Exception Returning other type than string or null
 */',
        'startLine' => 18,
        'endLine' => 18,
        'startColumn' => 9,
        'endColumn' => 36,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'Serializable',
        'implementingClassName' => 'Serializable',
        'currentClassName' => 'Serializable',
        'aliasName' => NULL,
      ),
      'unserialize' => 
      array (
        'name' => 'unserialize',
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'string\']',
                    'attributes' => 
                    array (
                      'startLine' => 26,
                      'endLine' => 26,
                      'startTokenPos' => 37,
                      'startFilePos' => 1002,
                      'endTokenPos' => 43,
                      'endFilePos' => 1020,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 26,
                      'endLine' => 26,
                      'startTokenPos' => 49,
                      'startFilePos' => 1032,
                      'endTokenPos' => 49,
                      'endFilePos' => 1033,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 26,
            'endLine' => 27,
            'startColumn' => 13,
            'endColumn' => 24,
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
 * Constructs the object.
 * @link https://php.net/manual/en/serializable.unserialize.php
 * @param string $data The string representation of the object.
 * @return void
 */',
        'startLine' => 25,
        'endLine' => 28,
        'startColumn' => 9,
        'endColumn' => 10,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'Serializable',
        'implementingClassName' => 'Serializable',
        'currentClassName' => 'Serializable',
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