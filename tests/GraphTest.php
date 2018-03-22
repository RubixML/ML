<?php

use Rubix\Engine\GraphNode;
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
                ['properties' => ['name' => 'Andrew', 'gender' => 'male', 'age' => 34]],
                ['properties' => ['name' => 'Julie', 'gender' => 'female', 'age' => 18]],
                ['properties' => ['name' => 'Frank', 'gender' => 'male', 'age' => 32]],
                ['properties' => ['name' => 'Seagal', 'gender' => 'male', 'age' => 49]],
                ['properties' => ['name' => 'Rich', 'gender' => 'male', 'age' => 62]],
                ['properties' => ['name' => 'Lacey', 'gender' => 'female', 'age' => 34]],
                ['properties' => ['name' => 'Steve', 'gender' => 'male', 'age' => 21]],
                ['properties' => ['name' => 'Harry', 'gender' => 'male', 'age' => 55]],
            ],
            'edges' => [
                ['start' => 'Andrew', 'end' => 'Rich', 'properties' => ['years' => 9]],
                ['start' => 'Andrew', 'end' => 'Lacey', 'properties' => ['years' => 5]],
                ['start' => 'Julie', 'end' => 'Frank', 'properties' => ['years' => 1]],
                ['start' => 'Julie', 'end' => 'Lacey', 'properties' => ['years' => 5]],
                ['start' => 'Frank', 'end' => 'Steve', 'properties' => ['years' => 1]],
                ['start' => 'Seagal', 'end' => 'Steve', 'properties' => ['years' => 4]],
                ['start' => 'Rich', 'end' => 'Andrew', 'properties' => ['years' => 9]],
                ['start' => 'Lacey', 'end' => 'Julie', 'properties' => ['years' => 5]],
                ['start' => 'Lacey', 'end' => 'Steve', 'properties' => ['years' => 6]],
                ['start' => 'Steve', 'end' => 'Seagal', 'properties' => ['years' => 4]],
                ['start' => 'Harry', 'end' => 'Frank', 'properties' => ['years' => 2]],
            ],
        ];

        $this->graph = new Graph();

        foreach ($data['nodes'] as $node) {
            $this->graph->insert($node['properties']);
        }

        foreach ($data['edges'] as $edge) {
            $this->graph->nodes()->where('name', '==', $edge['start'])->first()
                ->attach($this->graph->nodes()->where('name', '==', $edge['end'])->first(), $edge['properties']);
        }
    }

    public function test_build_graph()
    {
        $this->assertTrue($this->graph instanceof Graph);
    }

    public function test_graph_order()
    {
        $this->assertEquals(8, $this->graph->order());
    }

    public function test_graph_size()
    {
        $this->assertEquals(11, $this->graph->size());
    }

    public function test_insert_node()
    {
        $this->assertEquals(8, $this->graph->nodes()->count());

        $this->graph->insert([
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

    public function test_delete_node_from_graph()
    {
        $this->assertEquals(8, $this->graph->nodes()->count());
        $this->assertEquals('Seagal', $this->graph->find(4)->name);

        $this->graph->delete(4);

        $this->assertEquals(7, $this->graph->nodes()->count());

        $this->assertNull($this->graph->find(4));

        $this->assertNull($this->graph->find(7)->edges()->get(4));
    }

    public function test_find_path()
    {
        $path = $this->graph->findPath($this->graph->find(1), $this->graph->find(2));

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->length());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_all_paths()
    {
        $paths = $this->graph->findAllPaths($this->graph->find(6), $this->graph->find(7));

        $this->assertEquals(2, count($paths));

        $this->assertEquals(['Lacey', 'Julie', 'Frank', 'Steve'], $paths[0]->pluck('name'));
        $this->assertEquals(['Lacey', 'Steve'], $paths[1]->pluck('name'));
    }

    public function test_find_shortest_path()
    {
        $path = $this->graph->findShortestPath($this->graph->find(1), $this->graph->find(2));

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->length());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_all_pairs_shortest_paths()
    {
        $paths = $this->graph->findAllPairsShortestPaths();

        $this->assertEquals(27, count($paths));

        $this->assertEquals(['Andrew', 'Lacey', 'Julie'], $paths[0]->pluck('name'));
        $this->assertEquals(['Andrew', 'Lacey', 'Julie', 'Frank'], $paths[1]->pluck('name'));
        $this->assertEquals(['Andrew', 'Lacey', 'Steve', 'Seagal'], $paths[2]->pluck('name'));
        $this->assertEquals(['Andrew', 'Rich'], $paths[3]->pluck('name'));
    }

    public function test_find_shortest_weighted_path()
    {
        $path = $this->graph->findShortestWeightedPath($this->graph->find(1), $this->graph->find(2), 'years');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->length());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_shortest_unsigned_weighted_path()
    {
        $path = $this->graph->findShortestUnsignedWeightedPath($this->graph->find(1), $this->graph->find(2), 'years');

        $this->assertTrue($path instanceof Path);
        $this->assertEquals(3, $path->length());
        $this->assertEquals('Andrew', $path->first()->name);
        $this->assertEquals('Lacey', $path->next()->name);
        $this->assertEquals('Julie', $path->last()->name);
    }

    public function test_find_all_pairs_shortest_unsigned_weighted_path()
    {
        $paths = $this->graph->findAllPairsShortestUnsignedWeightedPath('years');

        $this->assertEquals(27, count($paths));

        $this->assertEquals(['Andrew', 'Lacey', 'Julie'], $paths[0]->pluck('name'));
        $this->assertEquals(['Andrew', 'Lacey', 'Julie', 'Frank'], $paths[1]->pluck('name'));
        $this->assertEquals(['Andrew', 'Lacey', 'Steve', 'Seagal'], $paths[2]->pluck('name'));
        $this->assertEquals(['Andrew', 'Rich'], $paths[3]->pluck('name'));
    }

    public function test_topological_sort()
    {
        $path = $this->graph->sort();

        $this->assertTrue(true);
    }

    public function test_graph_is_acyclic()
    {
        $this->assertFalse($this->graph->acyclic());
    }

    public function test_graph_is_cyclic()
    {
        $this->assertTrue($this->graph->cyclic());
    }
}
