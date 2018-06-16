<?php

namespace Rubix\ML\Graph;

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
     * Set a property on the object.
     *
     * @param  string  $property
     * @param  mixed  $value
     * @return void
     */
    public function set(string $property, $value = null) : void
    {
        $this->properties[$property] = $value;
    }

    /**
     * Remove a property from the object.
     *
     * @param  string  $property
     * @return self
     */
    public function remove(string $property) : void
    {
        if ($this->has($property)) {
            unset($this->properties[$property]);
        }
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
}
