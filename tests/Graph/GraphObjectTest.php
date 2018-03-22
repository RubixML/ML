<?php

use Rubix\Engine\Graph\GraphObject;
use PHPUnit\Framework\TestCase;

class GraphObjectTest extends TestCase
{
    protected $object;

    public function setUp()
    {
        $this->object = new GraphObject([
            'faces' => 6,
            'dimensions' => 3,
            'color' => 'red',
        ]);
    }

    public function test_create_object()
    {
        $this->assertTrue($this->object instanceof GraphObject);
    }

    public function test_object_has_property()
    {
        $this->assertTrue($this->object->has('color'));
        $this->assertTrue($this->object->has('faces'));
        $this->assertFalse($this->object->has('size'));
        $this->assertFalse($this->object->has(true));
    }

    public function test_get_properties()
    {
        $this->assertEquals(6, $this->object->faces);
        $this->assertEquals('red', $this->object->color);
        $this->assertEquals(null, $this->object->size);
        $this->assertEquals(3, $this->object->get('dimensions'));
    }

    public function test_set_properties()
    {
        $this->object->set('faces', 2);
        $this->object->set('dimensions', 2);
        $this->object->set('color', 'green');

        $this->assertEquals(2, $this->object->faces);
        $this->assertEquals(2, $this->object->dimensions);
        $this->assertEquals('green', $this->object->color);
    }

    public function test_update_properties()
    {
        $this->object->update([
            'color' => 'blue',
            'mass' => '100g',
        ]);

        $this->assertEquals('blue', $this->object->color);
        $this->assertEquals('100g', $this->object->mass);
        $this->assertEquals(6, $this->object->faces);
    }

    public function test_remove_property()
    {
        $this->assertTrue($this->object->has('color'));
        $this->assertTrue($this->object->has('dimensions'));
        $this->assertFalse($this->object->has('tacos'));

        $this->object->remove('color');

        $this->assertFalse($this->object->has('color'));
        $this->assertTrue($this->object->has('dimensions'));
        $this->assertFalse($this->object->has('tacos'));
    }
}
