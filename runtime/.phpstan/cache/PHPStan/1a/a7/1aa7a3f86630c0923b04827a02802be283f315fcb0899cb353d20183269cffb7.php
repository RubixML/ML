<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Serializers/Serializer.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Serializers\Serializer
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-7d1f61aeab5a36aa20bb3488d0b04a01f6377f6a12d18532dc41afc569c59ec8',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Serializers\\Serializer',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Serializers/Serializer.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Serializers',
    'name' => 'Rubix\\ML\\Serializers\\Serializer',
    'shortName' => 'Serializer',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Serializer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 16,
    'endLine' => 38,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Stringable',
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
            'startLine' => 26,
            'endLine' => 26,
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
        'startLine' => 26,
        'endLine' => 26,
        'startColumn' => 5,
        'endColumn' => 67,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\Serializer',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\Serializer',
        'currentClassName' => 'Rubix\\ML\\Serializers\\Serializer',
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
            'startLine' => 37,
            'endLine' => 37,
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
 * @throws \\Rubix\\ML\\Exceptions\\RuntimeException
 * @return Persistable
 */',
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 66,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Serializers\\Serializer',
        'implementingClassName' => 'Rubix\\ML\\Serializers\\Serializer',
        'currentClassName' => 'Rubix\\ML\\Serializers\\Serializer',
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