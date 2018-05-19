<?php

namespace Rubix\Engine\Estimators\Wrappers;

use Rubix\Engine\Estimators\Estimator;

abstract class Wrapper
{
    /**
     * The wrapped estimator instance.
     *
     * @var \Rubix\Engine\Estimators\Estimator
     */
    protected $estimator;

    /**
     * @param  \Rubix\Engine\Estimators\Estimator  $estimator
     * @return void
     */
    public function __construct(Estimator $estimator)
    {
        $this->estimator = $estimator;
    }

    /**
     * Return the underlying model instance.
     *
     * @return \Rubix\Engine\Estimators\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
