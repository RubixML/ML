<?php

namespace Rubix\ML\Exceptions;

use const Rubix\ML\VERSION;

class ClassRevisionMismatch extends RuntimeException
{
    /**
     * The version number of the library that the incompatible object was created with.
     *
     * @var string
     */
    protected $createdWithVersion;

    /**
     * @param string $createdWithVersion
     */
    public function __construct(string $createdWithVersion)
    {
        $direction = $createdWithVersion > VERSION ? 'up' : 'down';

        parent::__construct('Object serialized with incompatible class'
            . " revision, {$direction}grade to library version"
            . " $createdWithVersion and try again.");

        $this->createdWithVersion = $createdWithVersion;
    }

    /**
     * Return the version number of the library that the incompatible object was created with.
     *
     * @return string
     */
    public function createdWithVersion() : string
    {
        return $this->createdWithVersion;
    }
}
