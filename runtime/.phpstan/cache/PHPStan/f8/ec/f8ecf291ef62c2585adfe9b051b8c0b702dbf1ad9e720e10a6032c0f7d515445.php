<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionFunction-unserialize
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
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
        ),
        'startLine' => 47,
        'endLine' => 47,
        'startColumn' => 9,
        'endColumn' => 20,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'options' => 
      array (
        'name' => 'options',
        'default' => 
        array (
          'code' => '[]',
          'attributes' => 
          array (
            'startLine' => 49,
            'endLine' => 49,
            'startTokenPos' => 34,
            'startFilePos' => 2071,
            'endTokenPos' => 35,
            'endFilePos' => 2072,
          ),
        ),
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
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
            'name' => 'JetBrains\\PhpStorm\\Internal\\PhpStormStubsElementAvailable',
            'isRepeated' => false,
            'arguments' => 
            array (
              'from' => 
              array (
                'code' => '\'7.0\'',
                'attributes' => 
                array (
                  'startLine' => 48,
                  'endLine' => 48,
                  'startTokenPos' => 24,
                  'startFilePos' => 2038,
                  'endTokenPos' => 24,
                  'endFilePos' => 2042,
                ),
              ),
            ),
          ),
        ),
        'startLine' => 48,
        'endLine' => 49,
        'startColumn' => 9,
        'endColumn' => 27,
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
        'name' => 'mixed',
        'isIdentifier' => true,
      ),
    ),
    'attributes' => 
    array (
    ),
    'docComment' => '/**
 * Creates a PHP value from a stored representation
 * @link https://php.net/manual/en/function.unserialize.php
 * @param string $data <p>
 * The serialized string.
 * </p>
 * <p>
 * If the variable being unserialized is an object, after successfully
 * reconstructing the object PHP will automatically attempt to call the
 * __wakeup member function (if it exists).
 * </p>
 * <p>
 * unserialize_callback_func directive
 * </p>
 * <p>
 * It\'s possible to set a callback-function which will be called,
 * if an undefined class should be instantiated during unserializing.
 * (to prevent getting an incomplete object "__PHP_Incomplete_Class".)
 * Use your "php.ini", ini_set or ".htaccess"
 * to define \'unserialize_callback_func\'. Everytime an undefined class
 * should be instantiated, it\'ll be called. To disable this feature just
 * empty this setting.
 * </p>
 * @param array $options [optional]
 * <p>Any options to be provided to unserialize(), as an associative array.</p>
 * <p>
 * The \'allowed_classes\' option key may be set to a value that is
 * either an array of class names which should be accepted, FALSE to
 * accept no classes, or TRUE to accept all classes. If this option is defined
 * and unserialize() encounters an object of a class that isn\'t to be accepted,
 * then the object will be instantiated as __PHP_Incomplete_Class instead.
 * Omitting this option is the same as defining it as TRUE: PHP will attempt
 * to instantiate objects of any class.
 * </p>
 * @return mixed <p>The converted value is returned, and can be a boolean,
 * integer, float, string,
 * array or object.
 * </p>
 * <p>
 * In case the passed string is not unserializeable, false is returned and
 * E_NOTICE is issued.</p>
 */',
    'startLine' => 46,
    'endLine' => 52,
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
        'name' => 'unserialize',
        'filename' => 'phpstorm-stubs:standard/standard_4.stub',
        'extensionName' => 'standard',
        'aliasName' => NULL,
      ),
    ),
  ),
));