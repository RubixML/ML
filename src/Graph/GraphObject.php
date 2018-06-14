<?php

namespace Rubix\ML\Graph;

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
    public function set(string $property, $value = null) : GraphObject
    {
        $this->properties[$property] = $value;

        return $this;
    }

    /**
     * Update the object with properties from an associative array. O(P)
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
     * Return a given property or return default if not found.  If property is a
     * callback function, call and return it.
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
     * Remove a property from the object.
     *
     * @param  string  $property
     * @return self
     */
    public function remove(string $property) : self
    {
        if ($this->has($property)) {
            unset($this->properties[$property]);
        }

        return $this;
    }

    /**
     * Set a property using an assignment operation.
     *
     * @param  mixed  $property
     * @return mixed
     */
    public function __set(string $property, $value)
    {
        return $this->set($property, $value);
    }

    /**
     * Return a property value via object property accessor.
     *
     * @param  mixed  $property
     * @return mixed
     */
    public function __get(string $property)
    {
        return $this->get($property);
    }

    /**
     * Determine if a property is set, i.e. present and not null.
     *
     * @param  string  $property
     * @return bool
     */
    public function __isset(string $property)
    {
        return isset($this->properties[$property]);
    }

    /**
     * Remove a property from the object via unset().
     *
     * @param  string  $property
     * @return bool
     */
    public function __unset(string $property)
    {
        return $this->remove($property);
    }
}
