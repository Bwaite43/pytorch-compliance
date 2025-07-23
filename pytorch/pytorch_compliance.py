import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ComplianceDataset(Dataset):
    """PyTorch Dataset for compliance controls"""
    
    def __init__(self, df: pd.DataFrame, text_vectorizer=None, label_encoder=None, scaler=None, fit_transform=True):
        self.df = df.copy()
        self.fit_transform = fit_transform
        
        # Initialize preprocessing objects
        self.text_vectorizer = text_vectorizer or TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.label_encoder = label_encoder or LabelEncoder()
        self.scaler = scaler or StandardScaler()
        
        # Process the data
        self._prepare_features()
        
    def _prepare_features(self):
        """Extract and prepare features from controls data"""
        
        # 1. Text features using TF-IDF
        text_data = self.df['combined_text'].fillna('').astype(str)
        
        if self.fit_transform:
            self.text_features = self.text_vectorizer.fit_transform(text_data).toarray()
        else:
            self.text_features = self.text_vectorizer.transform(text_data).toarray()
        
        # 2. Categorical features
        categorical_features = []
        
        for _, row in self.df.iterrows():
            features = []
            
            # Control family one-hot encoding
            families = ['AC', 'AU', 'AT', 'CM', 'CP', 'IA', 'IR', 'MA', 'MP', 'PE', 'PL', 'PS', 'RA', 'CA', 'SC', 'SI', 'SA']
            family = row.get('control_family', 'Unknown')
            family_vector = [1 if f == family else 0 for f in families]
            features.extend(family_vector)
            
            # Framework encoding
            framework = row.get('framework', 'Unknown')
            features.append(1 if framework == 'GovRAMP' else 0)
            features.append(1 if framework == 'NIST' else 0)
            
            # Priority encoding
            priority_map = {'Low': 1, 'Moderate': 2, 'High': 3}
            priority = row.get('priority', 'Moderate')
            features.append(priority_map.get(priority, 2))
            
            # Text length features
            features.append(row.get('combined_text_length', 0) / 1000.0)  # Normalize
            
            categorical_features.append(features)
        
        categorical_features = np.array(categorical_features, dtype=np.float32)
        
        if self.fit_transform:
            self.categorical_features = self.scaler.fit_transform(categorical_features)
        else:
            self.categorical_features = self.scaler.transform(categorical_features)
        
        # 3. Combine all features
        self.features = np.concatenate([self.text_features, self.categorical_features], axis=1)
        
        # 4. Prepare labels (implementation status)
        status_labels = self.df['implementation_status'].fillna('Not Implemented')
        if self.fit_transform:
            self.labels = self.label_encoder.fit_transform(status_labels)
        else:
            self.labels = self.label_encoder.transform(status_labels)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

class ControlEmbeddingNetwork(nn.Module):
    """Neural network for creating control embeddings"""
    
    def __init__(self, input_size: int, embedding_dim: int = 128):
        super(ControlEmbeddingNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, embedding_dim),
            nn.Tanh()  # Normalize embeddings to [-1, 1]
        )
        
    def forward(self, x):
        return self.encoder(x)

class StatusPredictionNetwork(nn.Module):
    """Neural network for predicting implementation status"""
    
    def __init__(self, input_size: int, num_classes: int):
        super(StatusPredictionNetwork, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class PyTorchComplianceAnalyzer:
    """Main class for PyTorch-based compliance analysis"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 PyTorch Compliance Analyzer initialized on {self.device}")
        
        # Models
        self.embedding_model = None
        self.status_model = None
        
        # Data
        self.dataset = None
        self.control_embeddings = None
        self.controls_df = None
        
        # Results storage
        self.similarity_matrix = None
        self.analysis_results = {}
    
    def load_data(self, govramp_file: str, nist_file: str):
        """Load both GovRAMP and NIST control datasets"""
        
        print("📊 Loading control datasets...")
        
        # Load GovRAMP data
        govramp_df = pd.read_csv(govramp_file)
        print(f"✅ Loaded {len(govramp_df)} GovRAMP controls")
        
        # Load NIST data
        nist_df = pd.read_csv(nist_file)
        print(f"✅ Loaded {len(nist_df)} NIST controls")
        
        # Standardize NIST columns to match GovRAMP format
        nist_df = self._standardize_nist_columns(nist_df)
        
        # Combine datasets
        self.controls_df = pd.concat([govramp_df, nist_df], ignore_index=True)
        self.controls_df = self.controls_df.dropna(subset=['control_id']).reset_index(drop=True)
        
        print(f"🎯 Combined dataset: {len(self.controls_df)} total controls")
        print(f"📋 Frameworks: {self.controls_df['framework'].value_counts().to_dict()}")
        print(f"👥 Control families: {self.controls_df['control_family'].value_counts().head().to_dict()}")
        
        return self.controls_df
    
    def _standardize_nist_columns(self, nist_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize NIST DataFrame columns to match GovRAMP format"""
        
        print(f"🔧 Standardizing NIST columns...")
        print(f"📋 Original NIST columns: {list(nist_df.columns)}")
        
        # Create standardized dataframe
        nist_standardized = pd.DataFrame()
        
        # Map actual NIST columns to our standard format
        if 'identifier' in nist_df.columns:
            nist_standardized['control_id'] = nist_df['identifier']
        elif 'Control Identifier' in nist_df.columns:
            nist_standardized['control_id'] = nist_df['Control Identifier']
        else:
            # Use first column as fallback
            nist_standardized['control_id'] = nist_df.iloc[:, 0]
        
        # Description from name column
        if 'name' in nist_df.columns:
            nist_standardized['description'] = nist_df['name']
        elif 'Control Name' in nist_df.columns:
            nist_standardized['description'] = nist_df['Control Name']
        else:
            nist_standardized['description'] = 'NIST Control'
        
        # Guidance from control_text and discussion
        guidance_parts = []
        text_columns = ['control_text', 'discussion']
        
        for _, row in nist_df.iterrows():
            text_parts = []
            
            # Combine control_text and discussion
            for col in text_columns:
                if col in nist_df.columns and pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]).strip())
            
            # If no text found, use name/description
            if not text_parts:
                name_val = row.get('name', row.get('Control Name', ''))
                if pd.notna(name_val) and str(name_val).strip():
                    text_parts.append(str(name_val).strip())
            
            combined_text = ' '.join(text_parts) if text_parts else 'NIST Control'
            guidance_parts.append(combined_text)
        
        nist_standardized['guidance'] = guidance_parts
        nist_standardized['combined_text'] = guidance_parts
        
        # Add required columns with defaults
        nist_standardized['framework'] = 'NIST'
        nist_standardized['priority'] = 'Medium'
        nist_standardized['implementation_status'] = 'Not Implemented'
        
        # Extract control family from control_id (AC-1 -> AC)
        nist_standardized['control_family'] = nist_standardized['control_id'].apply(
            lambda x: x.split('-')[0] if isinstance(x, str) and '-' in x else 'Unknown'
        )
        
        # Text length
        nist_standardized['combined_text_length'] = nist_standardized['combined_text'].str.len()
        
        # Remove rows with missing control_id
        nist_standardized = nist_standardized.dropna(subset=['control_id'])
        nist_standardized = nist_standardized[nist_standardized['control_id'] != '']
        
        print(f"✅ Standardized {len(nist_standardized)} NIST controls")
        print(f"📊 NIST control families: {nist_standardized['control_family'].value_counts().head().to_dict()}")
        
        return nist_standardized
    
    def prepare_dataset(self):
        """Prepare PyTorch dataset from loaded data"""
        
        if self.controls_df is None:
            raise ValueError("No data loaded! Run load_data() first.")
        
        print("🔧 Preparing PyTorch dataset...")
        self.dataset = ComplianceDataset(self.controls_df)
        
        print(f"✅ Dataset prepared: {len(self.dataset)} samples")
        print(f"📊 Feature dimensions: {self.dataset.features.shape[1]}")
        print(f"🏷️  Classes: {list(self.dataset.label_encoder.classes_)}")
        
        return self.dataset
    
    def train_embedding_model(self, embedding_dim=128, epochs=100, batch_size=16, learning_rate=0.001):
        """Train the control embedding model"""
        
        if self.dataset is None:
            raise ValueError("Dataset not prepared! Run prepare_dataset() first.")
        
        print(f"🏋️  Training embedding model (dim={embedding_dim})...")
        
        # Initialize model
        input_size = self.dataset.features.shape[1]
        self.embedding_model = ControlEmbeddingNetwork(input_size, embedding_dim).to(self.device)
        
        # Data loader with drop_last=True to avoid batch size 1
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Optimizer
        optimizer = optim.Adam(self.embedding_model.parameters(), lr=learning_rate)
        
        # Training loop (using reconstruction loss for unsupervised learning)
        self.embedding_model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_features, _ in dataloader:
                batch_features = batch_features.to(self.device)
                
                # Get embeddings
                embeddings = self.embedding_model(batch_features)
                
                # Simple reconstruction loss (could be improved with contrastive learning)
                # For now, we encourage diverse embeddings with regularization
                loss = torch.mean(torch.sum(embeddings**2, dim=1))  # L2 regularization
                
                # Add diversity loss (encourage different embeddings for different controls)
                if len(embeddings) > 1:
                    similarity_matrix = torch.mm(embeddings, embeddings.t())
                    diversity_loss = torch.mean(torch.triu(similarity_matrix, diagonal=1)**2)
                    loss = loss + 0.1 * diversity_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        print("✅ Embedding model training complete!")
        
        # Generate embeddings for all controls
        self._generate_control_embeddings()
        
    def train_status_prediction_model(self, epochs=150, batch_size=32, learning_rate=0.001):
        """Train the implementation status prediction model"""
        
        if self.dataset is None:
            raise ValueError("Dataset not prepared! Run prepare_dataset() first.")
        
        print("🎯 Training status prediction model...")
        
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = self.dataset.features.shape[1]
        num_classes = len(self.dataset.label_encoder.classes_)
        self.status_model = StatusPredictionNetwork(input_size, num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.status_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.status_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.status_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            self.status_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.status_model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            scheduler.step(val_loss / len(val_loader))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.status_model.state_dict(), 'best_status_model.pth')
            
            if epoch % 25 == 0:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
                print(f'Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
                print('-' * 50)
        
        print(f"✅ Status prediction training complete! Best accuracy: {best_val_acc:.2f}%")
    
    def _generate_control_embeddings(self):
        """Generate embeddings for all controls"""
        
        if self.embedding_model is None:
            print("⚠️  No embedding model trained!")
            return
        
        print("📊 Generating control embeddings...")
        
        self.embedding_model.eval()
        embeddings = []
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(self.dataset.features).to(self.device)
            
            # Process in batches
            batch_size = 64
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i + batch_size]
                batch_embeddings = self.embedding_model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        self.control_embeddings = np.vstack(embeddings)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.control_embeddings)
        
        print(f"✅ Generated {len(self.control_embeddings)} control embeddings")
    
    def find_similar_controls(self, control_id: str, top_k: int = 10, cross_framework_only: bool = False):
        """Find controls similar to the specified control"""
        
        if self.control_embeddings is None:
            print("⚠️  No embeddings available! Train the embedding model first.")
            return []
        
        # Find target control
        target_idx = None
        for i, row in self.controls_df.iterrows():
            if row['control_id'] == control_id:
                target_idx = i
                break
        
        if target_idx is None:
            print(f"❌ Control {control_id} not found")
            return []
        
        target_control = self.controls_df.iloc[target_idx]
        similarities = self.similarity_matrix[target_idx]
        
        # Get top similar controls
        similar_indices = similarities.argsort()[::-1][1:top_k+10]  # Get extra in case we filter
        
        results = []
        for idx in similar_indices:
            if len(results) >= top_k:
                break
                
            similar_control = self.controls_df.iloc[idx]
            
            # Skip if same framework and cross_framework_only is True
            if cross_framework_only and similar_control['framework'] == target_control['framework']:
                continue
            
            results.append({
                'control_id': similar_control['control_id'],
                'framework': similar_control['framework'],
                'description': similar_control.get('description', '')[:100] + '...',
                'similarity_score': similarities[idx],
                'control_family': similar_control['control_family']
            })
        
        return results
    
    def create_framework_mapping(self, source_framework: str, target_framework: str, similarity_threshold: float = 0.6):
        """Create mapping between two frameworks"""
        
        print(f"🗺️  Creating mapping: {source_framework} → {target_framework}")
        
        source_controls = self.controls_df[self.controls_df['framework'] == source_framework]
        mapping = {}
        
        for _, control in source_controls.iterrows():
            similar_controls = self.find_similar_controls(
                control['control_id'], 
                top_k=5, 
                cross_framework_only=True
            )
            
            # Filter by target framework and similarity threshold
            mapped_controls = [
                ctrl for ctrl in similar_controls 
                if (ctrl['framework'] == target_framework and 
                    ctrl['similarity_score'] >= similarity_threshold)
            ]
            
            if mapped_controls:
                mapping[control['control_id']] = mapped_controls
        
        print(f"✅ Created {len(mapping)} mappings")
        return mapping
    
    def analyze_compliance_gaps(self, implemented_controls: List[str], target_framework: str = 'NIST'):
        """Analyze compliance gaps based on implemented controls"""
        
        print(f"🔍 Analyzing compliance gaps for {target_framework}...")
        
        # Get all target framework controls
        target_controls = set(
            self.controls_df[self.controls_df['framework'] == target_framework]['control_id']
        )
        
        # Find coverage through similarity
        covered_controls = set()
        coverage_details = {}
        
        for impl_control in implemented_controls:
            similar_controls = self.find_similar_controls(
                impl_control, 
                top_k=10, 
                cross_framework_only=True
            )
            
            for similar in similar_controls:
                if (similar['framework'] == target_framework and 
                    similar['similarity_score'] > 0.6):
                    
                    covered_controls.add(similar['control_id'])
                    
                    if similar['control_id'] not in coverage_details:
                        coverage_details[similar['control_id']] = []
                    
                    coverage_details[similar['control_id']].append({
                        'implemented_control': impl_control,
                        'similarity': similar['similarity_score']
                    })
        
        # Calculate gaps
        gap_controls = target_controls - covered_controls
        
        gap_analysis = {
            'total_target_controls': len(target_controls),
            'covered_controls': len(covered_controls),
            'gap_controls': len(gap_controls),
            'coverage_percentage': (len(covered_controls) / len(target_controls)) * 100,
            'gaps': list(gap_controls),
            'coverage_details': coverage_details
        }
        
        print(f"📊 Coverage: {gap_analysis['coverage_percentage']:.1f}%")
        print(f"❌ Gaps: {len(gap_controls)} controls need attention")
        
        return gap_analysis
    
    def predict_implementation_status(self, control_id: str):
        """Predict implementation status for a control"""
        
        if self.status_model is None:
            print("⚠️  No status prediction model trained!")
            return None
        
        # Find control
        control_idx = None
        for i, row in self.controls_df.iterrows():
            if row['control_id'] == control_id:
                control_idx = i
                break
        
        if control_idx is None:
            print(f"❌ Control {control_id} not found")
            return None
        
        self.status_model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(self.dataset.features[control_idx]).unsqueeze(0).to(self.device)
            outputs = self.status_model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_class = self.dataset.label_encoder.inverse_transform([predicted.item()])[0]
        confidence = probabilities[0][predicted.item()].item()
        
        # Get probability distribution
        prob_dist = {}
        for i, class_name in enumerate(self.dataset.label_encoder.classes_):
            prob_dist[class_name] = probabilities[0][i].item()
        
        return {
            'predicted_status': predicted_class,
            'confidence': confidence,
            'probability_distribution': prob_dist
        }
    
    def generate_compliance_report(self, implemented_controls: List[str] = None):
        """Generate comprehensive compliance analysis report"""
        
        print("📋 Generating compliance report...")
        
        report = {
            'summary': {
                'total_controls': len(self.controls_df),
                'frameworks': self.controls_df['framework'].value_counts().to_dict(),
                'control_families': self.controls_df['control_family'].value_counts().to_dict()
            },
            'framework_mappings': {},
            'gap_analysis': {},
            'recommendations': []
        }
        
        # Framework mappings
        frameworks = self.controls_df['framework'].unique()
        for i, fw1 in enumerate(frameworks):
            for fw2 in frameworks[i+1:]:
                mapping_key = f"{fw1}_to_{fw2}"
                mapping = self.create_framework_mapping(fw1, fw2)
                report['framework_mappings'][mapping_key] = {
                    'total_mappings': len(mapping),
                    'sample_mappings': dict(list(mapping.items())[:5])  # Show first 5
                }
        
        # Gap analysis if implemented controls provided
        if implemented_controls:
            for framework in frameworks:
                if framework != 'GovRAMP':  # Assuming GovRAMP is what's implemented
                    gaps = self.analyze_compliance_gaps(implemented_controls, framework)
                    report['gap_analysis'][framework] = gaps
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Save report
        with open('compliance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Report saved to compliance_report.json")
        return report
    
    def _generate_recommendations(self):
        """Generate intelligent recommendations"""
        
        recommendations = []
        
        # Find control families with high similarity across frameworks
        family_similarities = defaultdict(list)
        
        if self.similarity_matrix is not None:
            for i in range(len(self.controls_df)):
                for j in range(i+1, len(self.controls_df)):
                    if (self.controls_df.iloc[i]['framework'] != self.controls_df.iloc[j]['framework'] and
                        self.similarity_matrix[i][j] > 0.7):
                        
                        family1 = self.controls_df.iloc[i]['control_family']
                        family2 = self.controls_df.iloc[j]['control_family']
                        family_similarities[family1].append(family2)
        
        # Generate recommendations
        for family, related_families in family_similarities.items():
            if len(related_families) > 2:
                recommendations.append({
                    'type': 'cross_framework_alignment',
                    'message': f"Control family {family} has strong alignment across frameworks",
                    'related_families': list(set(related_families))
                })
        
        return recommendations[:10]  # Top 10 recommendations

def main():
    """Main execution function"""
    
    print("🚀 PyTorch Compliance Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PyTorchComplianceAnalyzer()
    
    # Load data
    analyzer.load_data('GovRAMPCoreControls_manual.csv', 'NIST_SP80053_rev5_catalog_load.csv')
    
    # Prepare dataset
    analyzer.prepare_dataset()
    
    # Train models
    print("\n🏋️  Training embedding model...")
    analyzer.train_embedding_model(embedding_dim=128, epochs=50)
    
    print("\n🎯 Training status prediction model...")
    analyzer.train_status_prediction_model(epochs=50)
    
    # Example analyses
    print("\n" + "="*60)
    print("📊 ANALYSIS EXAMPLES")
    print("="*60)
    
    # 1. Find similar controls
    print("\n🔍 Finding controls similar to AC-02:")  # Use actual control ID format
    similar_controls = analyzer.find_similar_controls('AC-02', top_k=5, cross_framework_only=True)
    for i, ctrl in enumerate(similar_controls, 1):
        print(f"  {i}. {ctrl['control_id']} ({ctrl['framework']}) - {ctrl['similarity_score']:.3f}")
        print(f"     {ctrl['description']}")
    
    # If AC-02 doesn't work, try the first available control
    if not similar_controls:
        print("  Trying first available GovRAMP control...")
        first_control = analyzer.controls_df[analyzer.controls_df['framework'] == 'GovRAMP']['control_id'].iloc[0]
        print(f"  Using {first_control} instead:")
        similar_controls = analyzer.find_similar_controls(first_control, top_k=5, cross_framework_only=True)
        for i, ctrl in enumerate(similar_controls, 1):
            print(f"  {i}. {ctrl['control_id']} ({ctrl['framework']}) - {ctrl['similarity_score']:.3f}")
            print(f"     {ctrl['description']}")

    
    # 2. Framework mapping
    print(f"\n🗺️  GovRAMP to NIST mapping (top 5):")
    mapping = analyzer.create_framework_mapping('GovRAMP', 'NIST', similarity_threshold=0.6)
    for i, (govramp_ctrl, nist_ctrls) in enumerate(list(mapping.items())[:5], 1):
        nist_ids = [ctrl['control_id'] for ctrl in nist_ctrls]
        print(f"  {i}. {govramp_ctrl} → {nist_ids}")
    
    # 3. Gap analysis example
    print(f"\n❌ Compliance gap analysis:")
    # Use actual control IDs from your data - check what's available first
    available_govramp_controls = analyzer.controls_df[
        analyzer.controls_df['framework'] == 'GovRAMP'
    ]['control_id'].tolist()[:10]  # Get first 10 actual controls
    
    print(f"  Available GovRAMP controls: {available_govramp_controls[:5]}...")
    
    gaps = analyzer.analyze_compliance_gaps(available_govramp_controls, 'NIST')
    print(f"  Coverage: {gaps['coverage_percentage']:.1f}%")
    print(f"  Missing controls: {gaps['gaps'][:5] if gaps['gaps'] else 'None'}...")  # Show first 5
    
    # 4. Status prediction example
    print(f"\n🎯 Status prediction:")
    # Use an actual control ID from the dataset
    first_control = analyzer.controls_df[analyzer.controls_df['framework'] == 'GovRAMP']['control_id'].iloc[0]
    print(f"  Predicting status for {first_control}:")
    prediction = analyzer.predict_implementation_status(first_control)
    if prediction:
        print(f"  Predicted: {prediction['predicted_status']} ({prediction['confidence']:.2f} confidence)")
        print(f"  Probability distribution:")
        for status, prob in prediction['probability_distribution'].items():
            print(f"    {status}: {prob:.3f}")
    else:
        print("  Prediction failed")
    
    # 5. Generate full report
    print(f"\n📋 Generating comprehensive report...")
    report = analyzer.generate_compliance_report(available_govramp_controls)
    
    print("\n✅ Analysis complete!")
    print("📄 Files generated:")
    print("  - best_embedding_model.pth")
    print("  - best_status_model.pth") 
    print("  - compliance_report.json")

if __name__ == "__main__":
    main()