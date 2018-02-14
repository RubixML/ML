<?php

use Rubix\Engine\Node;
use Rubix\Engine\Edge;
use Rubix\Engine\Path;
use Rubix\Engine\Graph;
use PHPUnit\Framework\TestCase;

class GraphTest extends TestCase
{
    protected $graph;

    public function setUp()
    {
        $data = [
            'nodes' => [
                ['id' => 1, 'properties' => ['name' => 'Andrew', 'gender' => 'male', 'age' => 34]],
                ['id' => 2, 'properties' => ['name' => 'Julie', 'gender' => 'female', 'age' => 18]],
                ['id' => 3, 'properties' => ['name' => 'Frank', 'gender' => 'male', 'age' => 32]],
                ['id' => 4, 'properties' => ['name' => 'Seagal', 'gender' => 'male', 'age' => 49]],
                ['id' => 5, 'properties' => ['name' => 'Rich', 'gender' => 'male', 'age' => 62]],
                ['id' => 6, 'properties' => ['name' => 'Lacey', 'gender' => 'female', 'age' => 34]],
                ['id' => 7, 'properties' => ['name' => 'Steve', 'gender' => 'male', 'age' => 21]],
                ['id' => 8, 'properties' => ['name' => 'Harry', 'gender' => 'male', 'age' => 55]],
            ],
            'edges' => [
                ['start' => 1, 'end' => 5, 'properties' => ['years' => 9]],
                ['start' => 1, 'end' => 6, 'properties' => ['years' => 5]],
                ['start' => 2, 'end' => 3, 'properties' => ['years' => 1]],
                ['start' => 2, 'end' => 6, 'properties' => ['years' => 5]],
                ['start' => 4, 'end' => 7, 'properties' => ['years' => 4]],
                ['start' => 5, 'end' => 1, 'properties' => ['years' => 9]],
                ['start' => 6, 'end' => 2, 'properties' => ['years' => 5]],
                ['start' => 6, 'end' => 7, 'properties' => ['years' => 6]],
                ['start' => 7, 'end' => 4, 'properties' => ['years' => 4]],
                ['start' => 8, 'end' => 3, 'properties' => ['years' => 2]],
            ],
        ];

        $this->graph = Graph::build($data['nodes'], $data['edges']);
    }

    public function test_insert_node()
    {
        $this->assertEquals(8, $this->graph->nodes()->count());

        $this->graph->insert(9, [
            'name' => 'Saoirse',
            'gender' => 'female',
        ]);

        $this->assertEquals(9, $this->graph->nodes()->count());

        $node = $this->graph->find(9);

        $this->assertEquals('Saoirse', $node->name);
        $this->assertEquals('female', $node->gender);
    }

    public function test_find_node()
    {
        $this->assertEquals('Julie', $this->graph->find(2)->name);
        $this->assertEquals('Frank', $this->graph->find(3)->name);
    }

    public function test_find_many_nodes()
    {
        $nodes = $this->graph->findMany([1, 3, 4]);

        $this->assertEquals(3, count($nodes));
        $this->assertEquals('Andrew', $nodes[1]->name);
        $this->assertEquals('Frank', $nodes[3]->name);
        $this->assertEquals('Seagal', $nodes[4]->name);
    }

    public function test_find_path()
    {
        $path = $this->graph->findPath($this->graph->find(1), $this->graph->find(2));

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->count());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_shortest_path()
    {
        $path = $this->graph->findShortestPath($this->graph->find(1), $this->graph->find(2));

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->count());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_shortest_weighted_path()
    {
        $path = $this->graph->findShortestPath($this->graph->find(1), $this->graph->find(2), 'years');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->count());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_shortest_unsigned_weighted_path()
    {
        $path = $this->graph->findShortestUnsignedWeightedPath($this->graph->find(1), $this->graph->find(2), 'years');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->count());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_topological_sort()
    {
        $path = $this->graph->sort();

        $this->assertTrue(true);
    }

    public function test_delete_node_from_graph()
    {
        $this->assertEquals(8, $this->graph->nodes()->count());
        $this->assertEquals('Seagal', $this->graph->find(4)->name);

        $this->graph->delete(4);

        $this->assertEquals(7, $this->graph->nodes()->count());

        $this->assertNull($this->graph->find(4));

        $this->assertNull($this->graph->find(7)->edges()->get(4));
    }

    public function test_graph_is_acyclic()
    {
        $this->assertFalse($this->graph->acyclic('FRIENDS'));
    }

    public function test_graph_is_cyclic()
    {
        $this->assertTrue($this->graph->cyclic('FRIENDS'));
    }
}
