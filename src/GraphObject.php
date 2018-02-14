<?php

namespace Rubix\Engine;

use InvalidArgumentException;

class GraphObject
{
    /**
     * The properties of the object.
     *
     * @var array
     */
    protected $properties = [
        //
    ];

    /**
     * @param  array  $properties
     * @return void
     */
    public function __construct(array $properties = [])
    {
        $this->update($properties);
    }

    /**
     * @return array
     */
    public function properties() : array
    {
        return $this->properties;
    }

    /**
     * Check to see if a given property is set on the object. O(1)
     *
     * @param  string  $property
     * @return bool
     */
    public function has(string $property) : bool
    {
        return array_key_exists($property, $this->properties);
    }

    /**
     * Set a given property on the object. O(1)
     *
     * @param  string  $property
     * @param  mixed  $value
     * @return self
     */
    public function set(string $property, $value) : GraphObject
    {
        $this->properties[$property] = $value;

        return $this;
    }

    /**
     * Update the object with properties from an associative array. O(N)
     *
     * @param  array  $properties
     * @return self
     */
    public function update(array $properties = []) : GraphObject
    {
        foreach ($properties as $property => $value) {
            $this->set($property, $value);
        }

        return $this;
    }

    /**
     * Return a given property or return default if not found. O(1)
     * If property is a function, call it and return its output.
     *
     * @param  string  $property
     * @param  mixed|null  $default
     * @return mixed|null
     */
    public function get(string $property, $default = null)
    {
        $value = $this->properties[$property] ?? $default;

        if (is_callable($value)) {
            return $value($this);
        }

        return $value;
    }

    /**
     * Magic getters.
     *
     * @param  mixed  $property
     * @return mixed
     */
    public function __get(string $property)
    {
        return $this->get($property);
    }
}
