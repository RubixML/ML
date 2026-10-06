<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\RuntimeException;

use function phpversion;
use function version_compare;

/**
 * @internal
 */
class ExtensionMaximumVersion extends Specification
{
    /**
     * The name of the extension under consideration.
     *
     * @var string
     */
    protected string $name;

    /**
     * The maximum version of the extension.
     *
     * @var string
     */
    protected string $maxVersion;

    /**
     * Build a specification object with the given arguments.
     *
     * @param string $name
     * @param string $maxVersion
     * @return self
     */
    public static function with(string $name, string $maxVersion) : self
    {
        return new self($name, $maxVersion);
    }

    /**
     * @param string $name
     * @param string $maxVersion
     */
    public function __construct(string $name, string $maxVersion)
    {
        $this->name = $name;
        $this->maxVersion = $maxVersion;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws RuntimeException
     */
    public function check() : void
    {
        $version = phpversion($this->name);

        if (!$version) {
            throw new RuntimeException("Version number for {$this->name} not available.");
        }

        if (version_compare($version, $this->maxVersion, '>')) {
            throw new RuntimeException("The {$this->name} extension version must be"
                . " less than {$this->maxVersion}, $version given.");
        }
    }
}
