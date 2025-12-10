"""
Package Manifest Adapter for axiom-kg

Parses dependency manifests from multiple ecosystems:
- Node.js: package.json
- Python: requirements.txt, pyproject.toml
- Rust: Cargo.toml

Use cases:
- "What does this project depend on?"
- "Which projects share dependencies?"
- "Detect version conflicts across repos"
- "Build dependency graph for an organization"

Usage:
    adapter = PackageAdapter()
    nodes = adapter.parse("package.json")
    
    # Compare dependencies
    diff = adapter.compare_deps("repo_a/package.json", "repo_b/package.json")
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import re

from .base import FileAdapter, JSONAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Package ecosystems → type_ coordinates
ECOSYSTEM_TO_TYPE = {
    "npm": 1,
    "pypi": 2,
    "cargo": 3,
    "maven": 4,
    "nuget": 5,
    "rubygems": 6,
    "go": 7,
}

# Dependency types → subtype coordinates
DEP_TYPE_TO_SUBTYPE = {
    "runtime": 1,
    "dev": 2,
    "peer": 3,
    "optional": 4,
    "build": 5,
}


class PackageAdapter(FileAdapter):
    """
    Adapter for package manifests across ecosystems.
    
    Converts dependencies into axiom-kg coordinates:
    - Packages → Entity nodes (Major 1)
    - Dependencies → Relation edges
    - Versions → Property nodes (Major 3)
    """
    
    DOMAIN_NAME = "package"
    SUPPORTED_EXTENSIONS = [".json", ".txt", ".toml"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _detect_format(self, path: Path) -> str:
        """Detect manifest format from filename."""
        name = path.name.lower()
        
        if name == "package.json":
            return "npm"
        if name in ["requirements.txt", "requirements-dev.txt"]:
            return "pip"
        if name == "pyproject.toml":
            return "pyproject"
        if name == "cargo.toml":
            return "cargo"
        if name == "go.mod":
            return "go"
        if name == "pom.xml":
            return "maven"
        
        return "unknown"
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse package manifest into axiom-kg nodes.
        
        Auto-detects format from filename.
        """
        path = Path(source) if isinstance(source, str) else source
        format_type = self._detect_format(path)
        
        if format_type == "npm":
            return self._parse_npm(path)
        elif format_type == "pip":
            return self._parse_pip(path)
        elif format_type == "pyproject":
            return self._parse_pyproject(path)
        elif format_type == "cargo":
            return self._parse_cargo(path)
        else:
            raise ValueError(f"Unknown manifest format: {path.name}")
    
    def _parse_npm(self, path: Path) -> List[Node]:
        """Parse package.json."""
        import json
        data = json.loads(path.read_text())
        
        nodes = []
        project_name = data.get("name", path.parent.name)
        
        # Create project node
        project_id = self.create_id(1, ECOSYSTEM_TO_TYPE["npm"], 1)
        project_node = Node(
            id=project_id,
            label=project_name,
            metadata={
                "ecosystem": "npm",
                "version": data.get("version"),
                "description": data.get("description"),
                "type": "project",
            }
        )
        nodes.append(project_node)
        
        # Parse dependencies
        for dep_type, deps in [
            ("runtime", data.get("dependencies", {})),
            ("dev", data.get("devDependencies", {})),
            ("peer", data.get("peerDependencies", {})),
            ("optional", data.get("optionalDependencies", {})),
        ]:
            for name, version in deps.items():
                dep_id = self.create_id(1, ECOSYSTEM_TO_TYPE["npm"], DEP_TYPE_TO_SUBTYPE[dep_type])
                dep_node = Node(
                    id=dep_id,
                    label=name,
                    metadata={
                        "ecosystem": "npm",
                        "version_spec": version,
                        "dep_type": dep_type,
                        "parent_project": project_name,
                    }
                )
                nodes.append(dep_node)
                project_node.add_relation(RelationType.PART_OF, dep_node)
        
        return nodes
    
    def _parse_pip(self, path: Path) -> List[Node]:
        """Parse requirements.txt."""
        content = path.read_text()
        nodes = []
        
        project_name = path.parent.name
        is_dev = "dev" in path.name.lower()
        dep_type = "dev" if is_dev else "runtime"
        
        # Create project node
        project_id = self.create_id(1, ECOSYSTEM_TO_TYPE["pypi"], 1)
        project_node = Node(
            id=project_id,
            label=project_name,
            metadata={
                "ecosystem": "pypi",
                "type": "project",
            }
        )
        nodes.append(project_node)
        
        # Parse requirements
        for line in content.split("\n"):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            
            # Parse package spec
            match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', line)
            if match:
                name = match.group(1)
                version_spec = match.group(2).strip()
                
                dep_id = self.create_id(1, ECOSYSTEM_TO_TYPE["pypi"], DEP_TYPE_TO_SUBTYPE[dep_type])
                dep_node = Node(
                    id=dep_id,
                    label=name,
                    metadata={
                        "ecosystem": "pypi",
                        "version_spec": version_spec,
                        "dep_type": dep_type,
                        "parent_project": project_name,
                    }
                )
                nodes.append(dep_node)
                project_node.add_relation(RelationType.PART_OF, dep_node)
        
        return nodes
    
    def _parse_pyproject(self, path: Path) -> List[Node]:
        """Parse pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            try:
                import toml as tomllib
            except ImportError:
                raise ImportError("tomllib (Python 3.11+) or toml package required")
        
        content = path.read_text()
        if hasattr(tomllib, 'loads'):
            data = tomllib.loads(content)
        else:
            data = tomllib.load(path.open('rb'))
        
        nodes = []
        
        # Get project info
        project = data.get("project", {})
        project_name = project.get("name", path.parent.name)
        
        # Create project node
        project_id = self.create_id(1, ECOSYSTEM_TO_TYPE["pypi"], 1)
        project_node = Node(
            id=project_id,
            label=project_name,
            metadata={
                "ecosystem": "pypi",
                "version": project.get("version"),
                "description": project.get("description"),
                "type": "project",
            }
        )
        nodes.append(project_node)
        
        # Parse dependencies
        dependencies = project.get("dependencies", [])
        for dep in dependencies:
            # Parse "package>=1.0" format
            match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', dep)
            if match:
                name = match.group(1)
                version_spec = match.group(2).strip()
                
                dep_id = self.create_id(1, ECOSYSTEM_TO_TYPE["pypi"], DEP_TYPE_TO_SUBTYPE["runtime"])
                dep_node = Node(
                    id=dep_id,
                    label=name,
                    metadata={
                        "ecosystem": "pypi",
                        "version_spec": version_spec,
                        "dep_type": "runtime",
                        "parent_project": project_name,
                    }
                )
                nodes.append(dep_node)
                project_node.add_relation(RelationType.PART_OF, dep_node)
        
        # Parse optional/dev dependencies
        optional = project.get("optional-dependencies", {})
        for group, deps in optional.items():
            dep_type = "dev" if group in ["dev", "test", "testing"] else "optional"
            for dep in deps:
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', dep)
                if match:
                    name = match.group(1)
                    version_spec = match.group(2).strip()
                    
                    dep_id = self.create_id(1, ECOSYSTEM_TO_TYPE["pypi"], DEP_TYPE_TO_SUBTYPE[dep_type])
                    dep_node = Node(
                        id=dep_id,
                        label=name,
                        metadata={
                            "ecosystem": "pypi",
                            "version_spec": version_spec,
                            "dep_type": dep_type,
                            "dep_group": group,
                            "parent_project": project_name,
                        }
                    )
                    nodes.append(dep_node)
                    project_node.add_relation(RelationType.PART_OF, dep_node)
        
        return nodes
    
    def _parse_cargo(self, path: Path) -> List[Node]:
        """Parse Cargo.toml."""
        try:
            import tomllib
        except ImportError:
            try:
                import toml as tomllib
            except ImportError:
                raise ImportError("tomllib (Python 3.11+) or toml package required")
        
        content = path.read_text()
        if hasattr(tomllib, 'loads'):
            data = tomllib.loads(content)
        else:
            data = tomllib.load(path.open('rb'))
        
        nodes = []
        
        # Get package info
        package = data.get("package", {})
        project_name = package.get("name", path.parent.name)
        
        # Create project node
        project_id = self.create_id(1, ECOSYSTEM_TO_TYPE["cargo"], 1)
        project_node = Node(
            id=project_id,
            label=project_name,
            metadata={
                "ecosystem": "cargo",
                "version": package.get("version"),
                "description": package.get("description"),
                "type": "project",
            }
        )
        nodes.append(project_node)
        
        # Parse dependencies
        for dep_type, deps in [
            ("runtime", data.get("dependencies", {})),
            ("dev", data.get("dev-dependencies", {})),
            ("build", data.get("build-dependencies", {})),
        ]:
            for name, spec in deps.items():
                # Handle both string and table specs
                if isinstance(spec, str):
                    version_spec = spec
                elif isinstance(spec, dict):
                    version_spec = spec.get("version", "*")
                else:
                    version_spec = str(spec)
                
                dep_id = self.create_id(1, ECOSYSTEM_TO_TYPE["cargo"], DEP_TYPE_TO_SUBTYPE[dep_type])
                dep_node = Node(
                    id=dep_id,
                    label=name,
                    metadata={
                        "ecosystem": "cargo",
                        "version_spec": version_spec,
                        "dep_type": dep_type,
                        "parent_project": project_name,
                    }
                )
                nodes.append(dep_node)
                project_node.add_relation(RelationType.PART_OF, dep_node)
        
        return nodes
    
    def compare_deps(self, manifest_a: Any, manifest_b: Any) -> Dict[str, Any]:
        """
        Compare dependencies between two projects.
        """
        nodes_a = self.parse(manifest_a)
        nodes_b = self.parse(manifest_b)
        
        # Get dependency names (exclude project nodes)
        deps_a = {n.label: n for n in nodes_a if n.metadata.get("type") != "project"}
        deps_b = {n.label: n for n in nodes_b if n.metadata.get("type") != "project"}
        
        shared = set(deps_a.keys()) & set(deps_b.keys())
        only_a = set(deps_a.keys()) - set(deps_b.keys())
        only_b = set(deps_b.keys()) - set(deps_a.keys())
        
        # Check version differences
        version_diffs = []
        for name in shared:
            ver_a = deps_a[name].metadata.get("version_spec")
            ver_b = deps_b[name].metadata.get("version_spec")
            if ver_a != ver_b:
                version_diffs.append({
                    "package": name,
                    "version_a": ver_a,
                    "version_b": ver_b,
                })
        
        return {
            "shared_deps": list(shared),
            "only_in_a": list(only_a),
            "only_in_b": list(only_b),
            "version_differences": version_diffs,
            "dep_count_a": len(deps_a),
            "dep_count_b": len(deps_b),
            "overlap_ratio": len(shared) / len(deps_a | deps_b) if deps_a or deps_b else 0,
        }
    
    def build_dependency_graph(self, manifests: List[Any]) -> Dict[str, Any]:
        """
        Build organization-wide dependency graph.
        """
        all_projects = []
        all_deps = {}  # dep_name -> list of projects using it
        
        for manifest in manifests:
            nodes = self.parse(manifest)
            
            project = None
            for n in nodes:
                if n.metadata.get("type") == "project":
                    project = n.label
                    all_projects.append(project)
                else:
                    dep_name = n.label
                    if dep_name not in all_deps:
                        all_deps[dep_name] = []
                    all_deps[dep_name].append({
                        "project": project,
                        "version": n.metadata.get("version_spec"),
                        "type": n.metadata.get("dep_type"),
                    })
        
        # Find most common deps
        common_deps = sorted(
            all_deps.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]
        
        return {
            "projects": all_projects,
            "unique_deps": len(all_deps),
            "total_dep_usages": sum(len(v) for v in all_deps.values()),
            "most_common": [
                {"name": name, "used_by": len(projects), "projects": [p["project"] for p in projects]}
                for name, projects in common_deps
            ],
        }
