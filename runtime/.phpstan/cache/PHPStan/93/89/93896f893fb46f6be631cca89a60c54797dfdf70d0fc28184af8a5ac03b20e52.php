<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionFunction-trigger_error
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'name' => 'trigger_error',
    'parameters' => 
    array (
      'message' => 
      array (
        'name' => 'message',
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
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 28,
        'endColumn' => 42,
        'parameterIndex' => 0,
        'isOptional' => false,
      ),
      'error_level' => 
      array (
        'name' => 'error_level',
        'default' => 
        array (
          'code' => '\\E_USER_NOTICE',
          'attributes' => 
          array (
            'startLine' => 20,
            'endLine' => 20,
            'startTokenPos' => 42,
            'startFilePos' => 849,
            'endTokenPos' => 42,
            'endFilePos' => 861,
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
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 45,
        'endColumn' => 76,
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
        'name' => 'true',
        'isIdentifier' => true,
      ),
    ),
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
            'code' => '[\'8.4\' => \'true\']',
            'attributes' => 
            array (
              'startLine' => 19,
              'endLine' => 19,
              'startTokenPos' => 11,
              'startFilePos' => 749,
              'endTokenPos' => 17,
              'endFilePos' => 765,
            ),
          ),
          'default' => 
          array (
            'code' => '\'bool\'',
            'attributes' => 
            array (
              'startLine' => 19,
              'endLine' => 19,
              'startTokenPos' => 23,
              'startFilePos' => 777,
              'endTokenPos' => 23,
              'endFilePos' => 782,
            ),
          ),
        ),
      ),
    ),
    'docComment' => '/**
 * Generates a user-level error/warning/notice message
 * @link https://php.net/manual/en/function.trigger-error.php
 * @param string $message <p>
 * The designated error message for this error. It\'s limited to 1024
 * characters in length. Any additional characters beyond 1024 will be
 * truncated.
 * </p>
 * @param int $error_level [optional] <p>
 * The designated error type for this error. It only works with the E_USER
 * family of constants, and will default to <b>E_USER_NOTICE</b>.
 * </p>
 * @return bool This function returns false if wrong <i>error_type</i> is
 * specified, true otherwise.
 */',
    'startLine' => 19,
    'endLine' => 22,
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
        'name' => 'trigger_error',
        'filename' => 'phpstorm-stubs:Core/Core.stub',
        'extensionName' => 'Core',
        'aliasName' => NULL,
      ),
    ),
  ),
));