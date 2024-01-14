<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\MissingExtension;

use function extension_loaded;

/**
 * @internal
 */
class ExtensionIsLoaded extends Specification
{
    /**
     * The name of the extension under consideration.
     *
     * @var string
     */
    protected string $name;

    /**
     * Build a specification object with the given arguments.
     *
     * @param string $name
     * @return self
     */
    public static function with(string $name) : self
    {
        return new self($name);
    }

    /**
     * @param string $name
     */
    public function __construct(string $name)
    {
        $this->name = $name;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws MissingExtension
     */
    public function check() : void
    {
        if (!extension_loaded($this->name)) {
            throw new MissingExtension($this->name);
        }
    }
}
