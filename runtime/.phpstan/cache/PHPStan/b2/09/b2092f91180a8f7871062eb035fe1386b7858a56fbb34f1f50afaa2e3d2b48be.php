<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Persisters/Filesystem.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Persisters\Filesystem
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-dc5a653cef894f39f5c43f6aa47819c5d3438ce03accc5a0eaa1d8d88ebcabc5',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Persisters\\Filesystem',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Persisters/Filesystem.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Persisters',
    'name' => 'Rubix\\ML\\Persisters\\Filesystem',
    'shortName' => 'Filesystem',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Filesystem
 *
 * Filesystems are local or remote storage drives that are organized by files
 * and folders. The filesystem persister serializes models to a file at a
 * user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 30,
    'endLine' => 157,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Persisters\\Persister',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'HISTORY_EXT' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'name' => 'HISTORY_EXT',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '\'old\'',
          'attributes' => 
          array (
            'startLine' => 37,
            'endLine' => 37,
            'startTokenPos' => 103,
            'startFilePos' => 879,
            'endTokenPos' => 103,
            'endFilePos' => 883,
          ),
        ),
        'docComment' => '/**
 * The extension to give files created as part of a persistable\'s save history.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 37,
      ),
    ),
    'immediateProperties' => 
    array (
      'path' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'name' => 'path',
        'modifiers' => 2,
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
 * The path to the model file on the filesystem.
 *
 * @var string
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 27,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'history' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'name' => 'history',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * Should we keep a history of past saves?
 *
 * @var bool
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 28,
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
          'path' => 
          array (
            'name' => 'path',
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
            'startLine' => 58,
            'endLine' => 58,
            'startColumn' => 33,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'history' => 
          array (
            'name' => 'history',
            'default' => 
            array (
              'code' => 'false',
              'attributes' => 
              array (
                'startLine' => 58,
                'endLine' => 58,
                'startTokenPos' => 143,
                'startFilePos' => 1301,
                'endTokenPos' => 143,
                'endFilePos' => 1305,
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
            'startLine' => 58,
            'endLine' => 58,
            'startColumn' => 47,
            'endColumn' => 67,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param string $path
 * @param bool $history
 * @throws InvalidArgumentException
 */',
        'startLine' => 58,
        'endLine' => 70,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Persisters',
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'currentClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'aliasName' => NULL,
      ),
      'save' => 
      array (
        'name' => 'save',
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
            'startLine' => 78,
            'endLine' => 78,
            'startColumn' => 26,
            'endColumn' => 43,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Save an encoding.
 *
 * @param Encoding $encoding
 * @throws \\RuntimeException
 */',
        'startLine' => 78,
        'endLine' => 113,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Persisters',
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'currentClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'aliasName' => NULL,
      ),
      'load' => 
      array (
        'name' => 'load',
        'parameters' => 
        array (
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
 * Load a persisted encoding.
 *
 * @throws \\RuntimeException
 * @return Encoding
 */',
        'startLine' => 121,
        'endLine' => 144,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Persisters',
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'currentClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
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
        'startLine' => 153,
        'endLine' => 156,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Persisters',
        'declaringClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'implementingClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
        'currentClassName' => 'Rubix\\ML\\Persisters\\Filesystem',
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