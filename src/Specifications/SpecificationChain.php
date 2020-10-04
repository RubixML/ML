<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

class SpecificationChain extends Specification
{
    /**
     * A list of specifications to check in order.
     *
     * @var \Rubix\ML\Specifications\Specification[]
     */
    protected $specifications;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Specifications\Specification[] $specifications
     * @return self
     */
    public static function with(array $specifications) : self
    {
        return new self($specifications);
    }

    /**
     * @param \Rubix\ML\Specifications\Specification[] $specifications
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $specifications)
    {
        foreach ($specifications as $specification) {
            if (!$specification instanceof Specification) {
                throw new InvalidArgumentException('Invalid specification.');
            }
        }

        $this->specifications = $specifications;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        foreach ($this->specifications as $specification) {
            $specification->check();
        }
    }
}
