<?php declare(strict_types = 1);

// phpinternal-PHPStan\BetterReflection\Reflection\ReflectionClass-pdostatement
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-dev-master@709e512-8.4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\InternalLocatedSource',
      'data' => 
      array (
        'name' => 'PDOStatement',
        'filename' => 'phpstorm-stubs:PDO/PDO.stub',
        'extensionName' => 'PDO',
        'aliasName' => NULL,
      ),
    ),
    'namespace' => NULL,
    'name' => 'PDOStatement',
    'shortName' => 'PDOStatement',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 1.0.0)<br/>
 * Represents a prepared statement and, after the statement is executed, an
 * associated result set.
 * @link https://php.net/manual/en/class.pdostatement.php
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 16,
    'endLine' => 568,
    'startColumn' => 5,
    'endColumn' => 5,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'IteratorAggregate',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'queryString' => 
      array (
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'name' => 'queryString',
        'modifiers' => 1,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * @var string
 */',
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
                'code' => '[\'8.1\' => \'string\']',
                'attributes' => 
                array (
                  'startLine' => 21,
                  'endLine' => 21,
                  'startTokenPos' => 53,
                  'startFilePos' => 725,
                  'endTokenPos' => 59,
                  'endFilePos' => 743,
                ),
              ),
              'default' => 
              array (
                'code' => '\'\'',
                'attributes' => 
                array (
                  'startLine' => 21,
                  'endLine' => 21,
                  'startTokenPos' => 65,
                  'startFilePos' => 755,
                  'endTokenPos' => 65,
                  'endFilePos' => 756,
                ),
              ),
            ),
          ),
        ),
        'startLine' => 21,
        'endLine' => 22,
        'startColumn' => 9,
        'endColumn' => 35,
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
      'execute' => 
      array (
        'name' => 'execute',
        'parameters' => 
        array (
          'params' => 
          array (
            'name' => 'params',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 50,
                'endLine' => 50,
                'startTokenPos' => 116,
                'startFilePos' => 2157,
                'endTokenPos' => 116,
                'endFilePos' => 2160,
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
                      'name' => 'array',
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'array|null\']',
                    'attributes' => 
                    array (
                      'startLine' => 49,
                      'endLine' => 49,
                      'startTokenPos' => 92,
                      'startFilePos' => 2085,
                      'endTokenPos' => 98,
                      'endFilePos' => 2107,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 49,
                      'endLine' => 49,
                      'startTokenPos' => 104,
                      'startFilePos' => 2119,
                      'endTokenPos' => 104,
                      'endFilePos' => 2120,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 49,
            'endLine' => 50,
            'startColumn' => 13,
            'endColumn' => 37,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Executes a prepared statement
 * @link https://php.net/manual/en/pdostatement.execute.php
 * @param array $params [optional] <p>
 * An array of values with as many elements as there are bound
 * parameters in the SQL statement being executed.
 * All values are treated as <b>PDO::PARAM_STR</b>.
 * </p>
 * <p>
 * You cannot bind multiple values to a single parameter; for example,
 * you cannot bind two values to a single named parameter in an IN()
 * clause.
 * </p>
 * <p>
 * You cannot bind more values than specified; if more keys exist in
 * <i>input_parameters</i> than in the SQL specified
 * in the <b>PDO::prepare</b>, then the statement will
 * fail and an error is emitted.
 * </p>
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 47,
        'endLine' => 53,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'fetch' => 
      array (
        'name' => 'fetch',
        'parameters' => 
        array (
          'mode' => 
          array (
            'name' => 'mode',
            'default' => 
            array (
              'code' => '\\PDO::FETCH_DEFAULT',
              'attributes' => 
              array (
                'startLine' => 87,
                'endLine' => 87,
                'startTokenPos' => 165,
                'startFilePos' => 3958,
                'endTokenPos' => 167,
                'endFilePos' => 3975,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 86,
                      'endLine' => 86,
                      'startTokenPos' => 143,
                      'startFilePos' => 3902,
                      'endTokenPos' => 149,
                      'endFilePos' => 3917,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 86,
                      'endLine' => 86,
                      'startTokenPos' => 155,
                      'startFilePos' => 3929,
                      'endTokenPos' => 155,
                      'endFilePos' => 3930,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 86,
            'endLine' => 87,
            'startColumn' => 13,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'cursorOrientation' => 
          array (
            'name' => 'cursorOrientation',
            'default' => 
            array (
              'code' => '\\PDO::FETCH_ORI_NEXT',
              'attributes' => 
              array (
                'startLine' => 89,
                'endLine' => 89,
                'startTokenPos' => 195,
                'startFilePos' => 4113,
                'endTokenPos' => 197,
                'endFilePos' => 4131,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 88,
                      'endLine' => 88,
                      'startTokenPos' => 173,
                      'startFilePos' => 4044,
                      'endTokenPos' => 179,
                      'endFilePos' => 4059,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 88,
                      'endLine' => 88,
                      'startTokenPos' => 185,
                      'startFilePos' => 4071,
                      'endTokenPos' => 185,
                      'endFilePos' => 4072,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 88,
            'endLine' => 89,
            'startColumn' => 13,
            'endColumn' => 56,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
          'cursorOffset' => 
          array (
            'name' => 'cursorOffset',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 91,
                'endLine' => 91,
                'startTokenPos' => 225,
                'startFilePos' => 4264,
                'endTokenPos' => 225,
                'endFilePos' => 4264,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 90,
                      'endLine' => 90,
                      'startTokenPos' => 203,
                      'startFilePos' => 4200,
                      'endTokenPos' => 209,
                      'endFilePos' => 4215,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 90,
                      'endLine' => 90,
                      'startTokenPos' => 215,
                      'startFilePos' => 4227,
                      'endTokenPos' => 215,
                      'endFilePos' => 4228,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 90,
            'endLine' => 91,
            'startColumn' => 13,
            'endColumn' => 33,
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
            'name' => 'mixed',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Fetches the next row from a result set
 * @link https://php.net/manual/en/pdostatement.fetch.php
 * @param int $mode [optional] <p>
 * Controls how the next row will be returned to the caller. This value
 * must be one of the PDO::FETCH_* constants,
 * defaulting to value of PDO::ATTR_DEFAULT_FETCH_MODE
 * (which defaults to PDO::FETCH_BOTH).
 * </p>
 * <p>
 * PDO::FETCH_ASSOC: returns an array indexed by column
 * name as returned in your result set
 * </p>
 * @param int $cursorOrientation [optional] <p>
 * For a PDOStatement object representing a scrollable cursor, this
 * value determines which row will be returned to the caller. This value
 * must be one of the PDO::FETCH_ORI_* constants,
 * defaulting to PDO::FETCH_ORI_NEXT. To request a
 * scrollable cursor for your PDOStatement object, you must set the
 * PDO::ATTR_CURSOR attribute to
 * PDO::CURSOR_SCROLL when you prepare the SQL
 * statement with <b>PDO::prepare</b>.
 * </p>
 * @param int $cursorOffset [optional]
 * @return mixed The return value of this function on success depends on the fetch type. In
 * all cases, <b>FALSE</b> is returned on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 84,
        'endLine' => 94,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'bindParam' => 
      array (
        'name' => 'bindParam',
        'parameters' => 
        array (
          'param' => 
          array (
            'name' => 'param',
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
                      'name' => 'int',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'string',
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
                    'code' => '[\'8.0\' => \'int|string\']',
                    'attributes' => 
                    array (
                      'startLine' => 129,
                      'endLine' => 129,
                      'startTokenPos' => 252,
                      'startFilePos' => 5995,
                      'endTokenPos' => 258,
                      'endFilePos' => 6017,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 129,
                      'endLine' => 129,
                      'startTokenPos' => 264,
                      'startFilePos' => 6029,
                      'endTokenPos' => 264,
                      'endFilePos' => 6030,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 129,
            'endLine' => 130,
            'startColumn' => 13,
            'endColumn' => 29,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'var' => 
          array (
            'name' => 'var',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => true,
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 131,
                      'endLine' => 131,
                      'startTokenPos' => 278,
                      'startFilePos' => 6131,
                      'endTokenPos' => 284,
                      'endFilePos' => 6148,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 131,
                      'endLine' => 131,
                      'startTokenPos' => 290,
                      'startFilePos' => 6160,
                      'endTokenPos' => 290,
                      'endFilePos' => 6161,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 131,
            'endLine' => 132,
            'startColumn' => 13,
            'endColumn' => 23,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'type' => 
          array (
            'name' => 'type',
            'default' => 
            array (
              'code' => '\\PDO::PARAM_STR',
              'attributes' => 
              array (
                'startLine' => 134,
                'endLine' => 134,
                'startTokenPos' => 325,
                'startFilePos' => 6312,
                'endTokenPos' => 327,
                'endFilePos' => 6325,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 133,
                      'endLine' => 133,
                      'startTokenPos' => 303,
                      'startFilePos' => 6256,
                      'endTokenPos' => 309,
                      'endFilePos' => 6271,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 133,
                      'endLine' => 133,
                      'startTokenPos' => 315,
                      'startFilePos' => 6283,
                      'endTokenPos' => 315,
                      'endFilePos' => 6284,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 133,
            'endLine' => 134,
            'startColumn' => 13,
            'endColumn' => 38,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'maxLength' => 
          array (
            'name' => 'maxLength',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 136,
                'endLine' => 136,
                'startTokenPos' => 355,
                'startFilePos' => 6455,
                'endTokenPos' => 355,
                'endFilePos' => 6455,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 135,
                      'endLine' => 135,
                      'startTokenPos' => 333,
                      'startFilePos' => 6394,
                      'endTokenPos' => 339,
                      'endFilePos' => 6409,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 135,
                      'endLine' => 135,
                      'startTokenPos' => 345,
                      'startFilePos' => 6421,
                      'endTokenPos' => 345,
                      'endFilePos' => 6422,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 135,
            'endLine' => 136,
            'startColumn' => 13,
            'endColumn' => 30,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'driverOptions' => 
          array (
            'name' => 'driverOptions',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 138,
                'endLine' => 138,
                'startTokenPos' => 383,
                'startFilePos' => 6593,
                'endTokenPos' => 383,
                'endFilePos' => 6596,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 137,
                      'endLine' => 137,
                      'startTokenPos' => 361,
                      'startFilePos' => 6524,
                      'endTokenPos' => 367,
                      'endFilePos' => 6541,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 137,
                      'endLine' => 137,
                      'startTokenPos' => 373,
                      'startFilePos' => 6553,
                      'endTokenPos' => 373,
                      'endFilePos' => 6554,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 137,
            'endLine' => 138,
            'startColumn' => 13,
            'endColumn' => 39,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Binds a parameter to the specified variable name
 * @link https://php.net/manual/en/pdostatement.bindparam.php
 * @param mixed $param <p>
 * Parameter identifier. For a prepared statement using named
 * placeholders, this will be a parameter name of the form
 * :name. For a prepared statement using
 * question mark placeholders, this will be the 1-indexed position of
 * the parameter.
 * </p>
 * @param mixed &$var <p>
 * Name of the PHP variable to bind to the SQL statement parameter.
 * </p>
 * @param int $type [optional] <p>
 * Explicit data type for the parameter using the PDO::PARAM_*
 * constants.
 * To return an INOUT parameter from a stored procedure,
 * use the bitwise OR operator to set the PDO::PARAM_INPUT_OUTPUT bits
 * for the <i>data_type</i> parameter.
 * </p>
 * @param int $maxLength [optional] <p>
 * Length of the data type. To indicate that a parameter is an OUT
 * parameter from a stored procedure, you must explicitly set the
 * length.
 * </p>
 * @param mixed $driverOptions [optional] <p>
 * </p>
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 127,
        'endLine' => 141,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'bindColumn' => 
      array (
        'name' => 'bindColumn',
        'parameters' => 
        array (
          'column' => 
          array (
            'name' => 'column',
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
                      'name' => 'int',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'string',
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
                    'code' => '[\'8.0\' => \'int|string\']',
                    'attributes' => 
                    array (
                      'startLine' => 169,
                      'endLine' => 169,
                      'startTokenPos' => 410,
                      'startFilePos' => 7952,
                      'endTokenPos' => 416,
                      'endFilePos' => 7974,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 169,
                      'endLine' => 169,
                      'startTokenPos' => 422,
                      'startFilePos' => 7986,
                      'endTokenPos' => 422,
                      'endFilePos' => 7987,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 169,
            'endLine' => 170,
            'startColumn' => 13,
            'endColumn' => 30,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'var' => 
          array (
            'name' => 'var',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => true,
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 171,
                      'endLine' => 171,
                      'startTokenPos' => 436,
                      'startFilePos' => 8089,
                      'endTokenPos' => 442,
                      'endFilePos' => 8106,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 171,
                      'endLine' => 171,
                      'startTokenPos' => 448,
                      'startFilePos' => 8118,
                      'endTokenPos' => 448,
                      'endFilePos' => 8119,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 171,
            'endLine' => 172,
            'startColumn' => 13,
            'endColumn' => 23,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'type' => 
          array (
            'name' => 'type',
            'default' => 
            array (
              'code' => '\\PDO::PARAM_STR',
              'attributes' => 
              array (
                'startLine' => 174,
                'endLine' => 174,
                'startTokenPos' => 483,
                'startFilePos' => 8270,
                'endTokenPos' => 485,
                'endFilePos' => 8283,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 173,
                      'endLine' => 173,
                      'startTokenPos' => 461,
                      'startFilePos' => 8214,
                      'endTokenPos' => 467,
                      'endFilePos' => 8229,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 173,
                      'endLine' => 173,
                      'startTokenPos' => 473,
                      'startFilePos' => 8241,
                      'endTokenPos' => 473,
                      'endFilePos' => 8242,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 173,
            'endLine' => 174,
            'startColumn' => 13,
            'endColumn' => 38,
            'parameterIndex' => 2,
            'isOptional' => true,
          ),
          'maxLength' => 
          array (
            'name' => 'maxLength',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 176,
                'endLine' => 176,
                'startTokenPos' => 513,
                'startFilePos' => 8413,
                'endTokenPos' => 513,
                'endFilePos' => 8413,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 175,
                      'endLine' => 175,
                      'startTokenPos' => 491,
                      'startFilePos' => 8352,
                      'endTokenPos' => 497,
                      'endFilePos' => 8367,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 175,
                      'endLine' => 175,
                      'startTokenPos' => 503,
                      'startFilePos' => 8379,
                      'endTokenPos' => 503,
                      'endFilePos' => 8380,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 175,
            'endLine' => 176,
            'startColumn' => 13,
            'endColumn' => 30,
            'parameterIndex' => 3,
            'isOptional' => true,
          ),
          'driverOptions' => 
          array (
            'name' => 'driverOptions',
            'default' => 
            array (
              'code' => '\\null',
              'attributes' => 
              array (
                'startLine' => 178,
                'endLine' => 178,
                'startTokenPos' => 541,
                'startFilePos' => 8551,
                'endTokenPos' => 541,
                'endFilePos' => 8554,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 177,
                      'endLine' => 177,
                      'startTokenPos' => 519,
                      'startFilePos' => 8482,
                      'endTokenPos' => 525,
                      'endFilePos' => 8499,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 177,
                      'endLine' => 177,
                      'startTokenPos' => 531,
                      'startFilePos' => 8511,
                      'endTokenPos' => 531,
                      'endFilePos' => 8512,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 177,
            'endLine' => 178,
            'startColumn' => 13,
            'endColumn' => 39,
            'parameterIndex' => 4,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Bind a column to a PHP variable
 * @link https://php.net/manual/en/pdostatement.bindcolumn.php
 * @param mixed $column <p>
 * Number of the column (1-indexed) or name of the column in the result set.
 * If using the column name, be aware that the name should match the
 * case of the column, as returned by the driver.
 * </p>
 * @param mixed &$var <p>
 * Name of the PHP variable to which the column will be bound.
 * </p>
 * @param int $type [optional] <p>
 * Data type of the parameter, specified by the PDO::PARAM_* constants.
 * </p>
 * @param int $maxLength [optional] <p>
 * A hint for pre-allocation.
 * </p>
 * @param mixed $driverOptions [optional] <p>
 * Optional parameter(s) for the driver.
 * </p>
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 167,
        'endLine' => 181,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'bindValue' => 
      array (
        'name' => 'bindValue',
        'parameters' => 
        array (
          'param' => 
          array (
            'name' => 'param',
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
                      'name' => 'int',
                      'isIdentifier' => true,
                    ),
                  ),
                  1 => 
                  array (
                    'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                    'data' => 
                    array (
                      'name' => 'string',
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
                    'code' => '[\'8.0\' => \'int|string\']',
                    'attributes' => 
                    array (
                      'startLine' => 206,
                      'endLine' => 206,
                      'startTokenPos' => 568,
                      'startFilePos' => 9743,
                      'endTokenPos' => 574,
                      'endFilePos' => 9765,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 206,
                      'endLine' => 206,
                      'startTokenPos' => 580,
                      'startFilePos' => 9777,
                      'endTokenPos' => 580,
                      'endFilePos' => 9778,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 206,
            'endLine' => 207,
            'startColumn' => 13,
            'endColumn' => 29,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'value' => 
          array (
            'name' => 'value',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 208,
                      'endLine' => 208,
                      'startTokenPos' => 594,
                      'startFilePos' => 9879,
                      'endTokenPos' => 600,
                      'endFilePos' => 9896,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 208,
                      'endLine' => 208,
                      'startTokenPos' => 606,
                      'startFilePos' => 9908,
                      'endTokenPos' => 606,
                      'endFilePos' => 9909,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 208,
            'endLine' => 209,
            'startColumn' => 13,
            'endColumn' => 24,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'type' => 
          array (
            'name' => 'type',
            'default' => 
            array (
              'code' => '\\PDO::PARAM_STR',
              'attributes' => 
              array (
                'startLine' => 211,
                'endLine' => 211,
                'startTokenPos' => 640,
                'startFilePos' => 10061,
                'endTokenPos' => 642,
                'endFilePos' => 10074,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 210,
                      'endLine' => 210,
                      'startTokenPos' => 618,
                      'startFilePos' => 10005,
                      'endTokenPos' => 624,
                      'endFilePos' => 10020,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 210,
                      'endLine' => 210,
                      'startTokenPos' => 630,
                      'startFilePos' => 10032,
                      'endTokenPos' => 630,
                      'endFilePos' => 10033,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 210,
            'endLine' => 211,
            'startColumn' => 13,
            'endColumn' => 38,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 1.0.0)<br/>
 * Binds a value to a parameter
 * @link https://php.net/manual/en/pdostatement.bindvalue.php
 * @param mixed $param <p>
 * Parameter identifier. For a prepared statement using named
 * placeholders, this will be a parameter name of the form
 * :name. For a prepared statement using
 * question mark placeholders, this will be the 1-indexed position of
 * the parameter.
 * </p>
 * @param mixed $value <p>
 * The value to bind to the parameter.
 * </p>
 * @param int $type [optional] <p>
 * Explicit data type for the parameter using the PDO::PARAM_*
 * constants.
 * </p>
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 204,
        'endLine' => 214,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'rowCount' => 
      array (
        'name' => 'rowCount',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Returns the number of rows affected by the last SQL statement
 * @link https://php.net/manual/en/pdostatement.rowcount.php
 * @return int the number of rows.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 223,
        'endLine' => 226,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'fetchColumn' => 
      array (
        'name' => 'fetchColumn',
        'parameters' => 
        array (
          'column' => 
          array (
            'name' => 'column',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 248,
                'endLine' => 248,
                'startTokenPos' => 712,
                'startFilePos' => 11755,
                'endTokenPos' => 712,
                'endFilePos' => 11755,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 247,
                      'endLine' => 247,
                      'startTokenPos' => 690,
                      'startFilePos' => 11697,
                      'endTokenPos' => 696,
                      'endFilePos' => 11712,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 247,
                      'endLine' => 247,
                      'startTokenPos' => 702,
                      'startFilePos' => 11724,
                      'endTokenPos' => 702,
                      'endFilePos' => 11725,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 247,
            'endLine' => 248,
            'startColumn' => 13,
            'endColumn' => 27,
            'parameterIndex' => 0,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.9.0)<br/>
 * Returns a single column from the next row of a result set
 * @link https://php.net/manual/en/pdostatement.fetchcolumn.php
 * @param int $column [optional] <p>
 * 0-indexed number of the column you wish to retrieve from the row. If
 * no value is supplied, <b>PDOStatement::fetchColumn</b>
 * fetches the first column.
 * </p>
 * @return mixed Returns a single column from the next row of a result
 * set or FALSE if there are no more rows.
 * </p>
 * <p>
 * There is no way to return another column from the same row if you
 * use <b>PDOStatement::fetchColumn</b> to retrieve data.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 245,
        'endLine' => 251,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'fetchAll' => 
      array (
        'name' => 'fetchAll',
        'parameters' => 
        array (
          'mode' => 
          array (
            'name' => 'mode',
            'default' => 
            array (
              'code' => '\\PDO::FETCH_DEFAULT',
              'attributes' => 
              array (
                'startLine' => 303,
                'endLine' => 303,
                'startTokenPos' => 761,
                'startFilePos' => 14452,
                'endTokenPos' => 763,
                'endFilePos' => 14469,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 302,
                      'endLine' => 302,
                      'startTokenPos' => 739,
                      'startFilePos' => 14396,
                      'endTokenPos' => 745,
                      'endFilePos' => 14411,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 302,
                      'endLine' => 302,
                      'startTokenPos' => 751,
                      'startFilePos' => 14423,
                      'endTokenPos' => 751,
                      'endFilePos' => 14424,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 302,
            'endLine' => 303,
            'startColumn' => 13,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'args' => 
          array (
            'name' => 'args',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => true,
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 304,
                      'endLine' => 304,
                      'startTokenPos' => 769,
                      'startFilePos' => 14538,
                      'endTokenPos' => 775,
                      'endFilePos' => 14555,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 304,
                      'endLine' => 304,
                      'startTokenPos' => 781,
                      'startFilePos' => 14567,
                      'endTokenPos' => 781,
                      'endFilePos' => 14568,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 304,
            'endLine' => 305,
            'startColumn' => 13,
            'endColumn' => 26,
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Returns an array containing all of the result set rows
 * @link https://php.net/manual/en/pdostatement.fetchall.php
 * @param int $mode [optional] <p>
 * Controls the contents of the returned array as documented in
 * <b>PDOStatement::fetch</b>.
 * Defaults to value of <b>PDO::ATTR_DEFAULT_FETCH_MODE</b>
 * (which defaults to <b>PDO::FETCH_BOTH</b>)
 * </p>
 * <p>
 * To return an array consisting of all values of a single column from
 * the result set, specify <b>PDO::FETCH_COLUMN</b>. You
 * can specify which column you want with the
 * <i>column-index</i> parameter.
 * </p>
 * <p>
 * To fetch only the unique values of a single column from the result set,
 * bitwise-OR <b>PDO::FETCH_COLUMN</b> with
 * <b>PDO::FETCH_UNIQUE</b>.
 * </p>
 * <p>
 * To return an associative array grouped by the values of a specified
 * column, bitwise-OR <b>PDO::FETCH_COLUMN</b> with
 * <b>PDO::FETCH_GROUP</b>.
 * </p>
 * @param mixed ...$args <p>
 * Arguments of custom class constructor when the <i>fetch_style</i>
 * parameter is <b>PDO::FETCH_CLASS</b>.
 * </p>
 * @return array <b>PDOStatement::fetchAll</b> returns an array containing
 * all of the remaining rows in the result set. The array represents each
 * row as either an array of column values or an object with properties
 * corresponding to each column name.
 * An empty array is returned if there are zero results to fetch.
 * </p>
 * @throws PDOException <b>PDOStatement::fetchAll</b> throws on failure if the
 * attribute <b>PDO::ATTR_ERRMODE</b> is set to <b>PDO::ERRMODE_EXCEPTION</b>.
 * </p>
 * <p>
 * Using this method to fetch large result sets will result in a heavy
 * demand on system and possibly network resources. Rather than retrieving
 * all of the data and manipulating it in PHP, consider using the database
 * server to manipulate the result sets. For example, use the WHERE and
 * ORDER BY clauses in SQL to restrict results before retrieving and
 * processing them with PHP.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 300,
        'endLine' => 308,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => true,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'fetchObject' => 
      array (
        'name' => 'fetchObject',
        'parameters' => 
        array (
          'class' => 
          array (
            'name' => 'class',
            'default' => 
            array (
              'code' => '"stdClass"',
              'attributes' => 
              array (
                'startLine' => 329,
                'endLine' => 329,
                'startTokenPos' => 839,
                'startFilePos' => 15664,
                'endTokenPos' => 839,
                'endFilePos' => 15673,
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
                      'name' => 'string',
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'string|null\']',
                    'attributes' => 
                    array (
                      'startLine' => 328,
                      'endLine' => 328,
                      'startTokenPos' => 815,
                      'startFilePos' => 15591,
                      'endTokenPos' => 821,
                      'endFilePos' => 15614,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 328,
                      'endLine' => 328,
                      'startTokenPos' => 827,
                      'startFilePos' => 15626,
                      'endTokenPos' => 827,
                      'endFilePos' => 15627,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 328,
            'endLine' => 329,
            'startColumn' => 13,
            'endColumn' => 43,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
          'constructorArgs' => 
          array (
            'name' => 'constructorArgs',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 331,
                'endLine' => 331,
                'startTokenPos' => 867,
                'startFilePos' => 15813,
                'endTokenPos' => 868,
                'endFilePos' => 15814,
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
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'array\']',
                    'attributes' => 
                    array (
                      'startLine' => 330,
                      'endLine' => 330,
                      'startTokenPos' => 845,
                      'startFilePos' => 15742,
                      'endTokenPos' => 851,
                      'endFilePos' => 15759,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 330,
                      'endLine' => 330,
                      'startTokenPos' => 857,
                      'startFilePos' => 15771,
                      'endTokenPos' => 857,
                      'endFilePos' => 15772,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 330,
            'endLine' => 331,
            'startColumn' => 13,
            'endColumn' => 39,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
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
                  'name' => 'object',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'false',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * @template T
 *
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.4)<br/>
 * Fetches the next row and returns it as an object.
 * @link https://php.net/manual/en/pdostatement.fetchobject.php
 * @param class-string<T> $class [optional] <p>
 * Name of the created class.
 * </p>
 * @param array $constructorArgs [optional] <p>
 * Elements of this array are passed to the constructor.
 * </p>
 * @return T|stdClass|null an instance of the required class with property names that
 * correspond to the column names or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 326,
        'endLine' => 334,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'errorCode' => 
      array (
        'name' => 'errorCode',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
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
                  'name' => 'string',
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
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Fetch the SQLSTATE associated with the last operation on the statement handle
 * @link https://php.net/manual/en/pdostatement.errorcode.php
 * @return string Identical to <b>PDO::errorCode</b>, except that
 * <b>PDOStatement::errorCode</b> only retrieves error codes
 * for operations performed with PDOStatement objects.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 344,
        'endLine' => 347,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'errorInfo' => 
      array (
        'name' => 'errorInfo',
        'parameters' => 
        array (
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\ArrayShape',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '[0 => "string", 1 => "int", 2 => "string"]',
                'attributes' => 
                array (
                  'startLine' => 374,
                  'endLine' => 374,
                  'startTokenPos' => 908,
                  'startFilePos' => 17591,
                  'endTokenPos' => 928,
                  'endFilePos' => 17632,
                ),
              ),
            ),
          ),
          1 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.1.0)<br/>
 * Fetch extended error information associated with the last operation on the statement handle
 * @link https://php.net/manual/en/pdostatement.errorinfo.php
 * @return array <b>PDOStatement::errorInfo</b> returns an array of
 * error information about the last operation performed by this
 * statement handle. The array consists of the following fields:
 * <tr valign="top">
 * <td>Element</td>
 * <td>Information</td>
 * </tr>
 * <tr valign="top">
 * <td>0</td>
 * <td>SQLSTATE error code (a five characters alphanumeric identifier defined
 * in the ANSI SQL standard).</td>
 * </tr>
 * <tr valign="top">
 * <td>1</td>
 * <td>Driver specific error code.</td>
 * </tr>
 * <tr valign="top">
 * <td>2</td>
 * <td>Driver specific error message.</td>
 * </tr>
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 374,
        'endLine' => 378,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'setAttribute' => 
      array (
        'name' => 'setAttribute',
        'parameters' => 
        array (
          'attribute' => 
          array (
            'name' => 'attribute',
            'default' => NULL,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 391,
                      'endLine' => 391,
                      'startTokenPos' => 967,
                      'startFilePos' => 18380,
                      'endTokenPos' => 973,
                      'endFilePos' => 18395,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 391,
                      'endLine' => 391,
                      'startTokenPos' => 979,
                      'startFilePos' => 18407,
                      'endTokenPos' => 979,
                      'endFilePos' => 18408,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 391,
            'endLine' => 392,
            'startColumn' => 13,
            'endColumn' => 26,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'value' => 
          array (
            'name' => 'value',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
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
                    'code' => '[\'8.0\' => \'mixed\']',
                    'attributes' => 
                    array (
                      'startLine' => 393,
                      'endLine' => 393,
                      'startTokenPos' => 991,
                      'startFilePos' => 18506,
                      'endTokenPos' => 997,
                      'endFilePos' => 18523,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 393,
                      'endLine' => 393,
                      'startTokenPos' => 1003,
                      'startFilePos' => 18535,
                      'endTokenPos' => 1003,
                      'endFilePos' => 18536,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 393,
            'endLine' => 394,
            'startColumn' => 13,
            'endColumn' => 24,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Set a statement attribute
 * @link https://php.net/manual/en/pdostatement.setattribute.php
 * @param int $attribute
 * @param mixed $value
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 389,
        'endLine' => 397,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'getAttribute' => 
      array (
        'name' => 'getAttribute',
        'parameters' => 
        array (
          'name' => 
          array (
            'name' => 'name',
            'default' => NULL,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 408,
                      'endLine' => 408,
                      'startTokenPos' => 1036,
                      'startFilePos' => 19085,
                      'endTokenPos' => 1042,
                      'endFilePos' => 19100,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 408,
                      'endLine' => 408,
                      'startTokenPos' => 1048,
                      'startFilePos' => 19112,
                      'endTokenPos' => 1048,
                      'endFilePos' => 19113,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 408,
            'endLine' => 409,
            'startColumn' => 13,
            'endColumn' => 21,
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
            'name' => 'mixed',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Retrieve a statement attribute
 * @link https://php.net/manual/en/pdostatement.getattribute.php
 * @param int $name
 * @return mixed the attribute value.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 406,
        'endLine' => 412,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'columnCount' => 
      array (
        'name' => 'columnCount',
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
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Returns the number of columns in the result set
 * @link https://php.net/manual/en/pdostatement.columncount.php
 * @return int the number of columns in the result set represented by the
 * PDOStatement object. If there is no result set,
 * <b>PDOStatement::columnCount</b> returns 0.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 423,
        'endLine' => 426,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'getColumnMeta' => 
      array (
        'name' => 'getColumnMeta',
        'parameters' => 
        array (
          'column' => 
          array (
            'name' => 'column',
            'default' => NULL,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 491,
                      'endLine' => 491,
                      'startTokenPos' => 1164,
                      'startFilePos' => 22532,
                      'endTokenPos' => 1170,
                      'endFilePos' => 22547,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 491,
                      'endLine' => 491,
                      'startTokenPos' => 1176,
                      'startFilePos' => 22559,
                      'endTokenPos' => 1176,
                      'endFilePos' => 22560,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 491,
            'endLine' => 492,
            'startColumn' => 13,
            'endColumn' => 23,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
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
                  'name' => 'array',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'false',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
          1 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\ArrayShape',
            'isRepeated' => false,
            'arguments' => 
            array (
              0 => 
              array (
                'code' => '["name" => "string", "len" => "int", "precision" => "int", "oci:decl_type" => "int|string", "native_type" => "string", "scale" => "int", "flags" => "array", "pdo_type" => "int"]',
                'attributes' => 
                array (
                  'startLine' => 489,
                  'endLine' => 489,
                  'startTokenPos' => 1095,
                  'startFilePos' => 22247,
                  'endTokenPos' => 1150,
                  'endFilePos' => 22423,
                ),
              ),
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Returns metadata for a column in a result set
 * @link https://php.net/manual/en/pdostatement.getcolumnmeta.php
 * @param int $column <p>
 * The 0-indexed column in the result set.
 * </p>
 * @return array|false an associative array containing the following values representing
 * the metadata for a single column:
 * </p>
 * <table>
 * Column metadata
 * <tr valign="top">
 * <td>Name</td>
 * <td>Value</td>
 * </tr>
 * <tr valign="top">
 * <td>native_type</td>
 * <td>The PHP native type used to represent the column value.</td>
 * </tr>
 * <tr valign="top">
 * <td>driver:decl_type</td>
 * <td>The SQL type used to represent the column value in the database.
 * If the column in the result set is the result of a function, this value
 * is not returned by <b>PDOStatement::getColumnMeta</b>.
 * </td>
 * </tr>
 * <tr valign="top">
 * <td>flags</td>
 * <td>Any flags set for this column.</td>
 * </tr>
 * <tr valign="top">
 * <td>name</td>
 * <td>The name of this column as returned by the database.</td>
 * </tr>
 * <tr valign="top">
 * <td>table</td>
 * <td>The name of this column\'s table as returned by the database.</td>
 * </tr>
 * <tr valign="top">
 * <td>len</td>
 * <td>The length of this column. Normally -1 for
 * types other than floating point decimals.</td>
 * </tr>
 * <tr valign="top">
 * <td>precision</td>
 * <td>The numeric precision of this column. Normally
 * 0 for types other than floating point
 * decimals.</td>
 * </tr>
 * <tr valign="top">
 * <td>pdo_type</td>
 * <td>The type of this column as represented by the
 * PDO::PARAM_* constants.</td>
 * </tr>
 * </table>
 * <p>
 * Returns <b>FALSE</b> if the requested column does not exist in the result set,
 * or if no result set exists.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 488,
        'endLine' => 495,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'setFetchMode' => 
      array (
        'name' => 'setFetchMode',
        'parameters' => 
        array (
          'mode' => 
          array (
            'name' => 'mode',
            'default' => NULL,
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
              0 => 
              array (
                'name' => 'JetBrains\\PhpStorm\\Internal\\LanguageLevelTypeAware',
                'isRepeated' => false,
                'arguments' => 
                array (
                  0 => 
                  array (
                    'code' => '[\'8.0\' => \'int\']',
                    'attributes' => 
                    array (
                      'startLine' => 511,
                      'endLine' => 511,
                      'startTokenPos' => 1230,
                      'startFilePos' => 23495,
                      'endTokenPos' => 1236,
                      'endFilePos' => 23510,
                    ),
                  ),
                  'default' => 
                  array (
                    'code' => '\'\'',
                    'attributes' => 
                    array (
                      'startLine' => 511,
                      'endLine' => 511,
                      'startTokenPos' => 1242,
                      'startFilePos' => 23522,
                      'endTokenPos' => 1242,
                      'endFilePos' => 23523,
                    ),
                  ),
                ),
              ),
            ),
            'startLine' => 511,
            'endLine' => 512,
            'startColumn' => 13,
            'endColumn' => 21,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'args' => 
          array (
            'name' => 'args',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => true,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 513,
            'endLine' => 513,
            'startColumn' => 13,
            'endColumn' => 26,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
          1 => 
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
                  'startLine' => 509,
                  'endLine' => 509,
                  'startTokenPos' => 1204,
                  'startFilePos' => 23354,
                  'endTokenPos' => 1210,
                  'endFilePos' => 23370,
                ),
              ),
              'default' => 
              array (
                'code' => '\'bool\'',
                'attributes' => 
                array (
                  'startLine' => 509,
                  'endLine' => 509,
                  'startTokenPos' => 1216,
                  'startFilePos' => 23382,
                  'endTokenPos' => 1216,
                  'endFilePos' => 23387,
                ),
              ),
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Set the default fetch mode for this statement
 * @link https://php.net/manual/en/pdostatement.setfetchmode.php
 * @param int $mode <p>
 * The fetch mode must be one of the PDO::FETCH_* constants.
 * </p>
 * @param mixed ...$args <p> Constructor arguments. </p>
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 508,
        'endLine' => 516,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => true,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'nextRowset' => 
      array (
        'name' => 'nextRowset',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.2.0)<br/>
 * Advances to the next rowset in a multi-rowset statement handle
 * @link https://php.net/manual/en/pdostatement.nextrowset.php
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 525,
        'endLine' => 528,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'closeCursor' => 
      array (
        'name' => 'closeCursor',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.9.0)<br/>
 * Closes the cursor, enabling the statement to be executed again.
 * @link https://php.net/manual/en/pdostatement.closecursor.php
 * @return bool <b>TRUE</b> on success or <b>FALSE</b> on failure.
 * @throws PDOException On error if PDO::ERRMODE_EXCEPTION option is true.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 537,
        'endLine' => 540,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'debugDumpParams' => 
      array (
        'name' => 'debugDumpParams',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
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
                  'name' => 'bool',
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
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'JetBrains\\PhpStorm\\Internal\\TentativeType',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * (PHP 5 &gt;= 5.1.0, PHP 7, PECL pdo &gt;= 0.9.0)<br/>
 * Dump an SQL prepared command
 * @link https://php.net/manual/en/pdostatement.debugdumpparams.php
 * @return bool|null No value is returned.
 * @betterReflectionTentativeReturnType
 */',
        'startLine' => 548,
        'endLine' => 551,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      '__wakeup' => 
      array (
        'name' => '__wakeup',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 552,
        'endLine' => 554,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 33,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      '__sleep' => 
      array (
        'name' => '__sleep',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 555,
        'endLine' => 557,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 33,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
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
            'name' => 'Iterator',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @return Iterator
 * @since 8.0
 */',
        'startLine' => 562,
        'endLine' => 564,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
        'aliasName' => NULL,
      ),
      'connect' => 
      array (
        'name' => 'connect',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => NULL,
        'startLine' => 565,
        'endLine' => 567,
        'startColumn' => 9,
        'endColumn' => 9,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => NULL,
        'declaringClassName' => 'PDOStatement',
        'implementingClassName' => 'PDOStatement',
        'currentClassName' => 'PDOStatement',
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